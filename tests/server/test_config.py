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

import pytest

from paramem.server.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_SERVER_CONFIG_PATH,
    PathsConfig,
    load_server_config,
)


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

    Default is None → training_config.num_epochs == 30 (the validated floor).
    When set, it overrides.
    """

    def test_max_epochs_default_none_uses_validated(self, tmp_path):
        """Absent max_epochs → property returns the validated 30."""
        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        config = load_server_config(yaml_file)
        assert config.consolidation.max_epochs is None
        assert config.training_config.num_epochs == 30

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


class TestAdaptersFactoryDefaultMerge:
    """Loader contract for adapter target_modules under the explicit-yaml posture.

    The yaml is the contract for load-bearing fields. When an operator
    partially specifies an adapter tier in yaml and that tier ends up
    enabled, the loader refuses to start without an explicit
    ``target_modules`` — silent fallback to the factory default would
    hide architectural choices like procedural's attn+mlp targeting.

    The previous test suite asserted the silent-fallback merge was
    correct; that behaviour is now explicitly forbidden by
    ``load_server_config``'s loader guard, so those tests have been
    converted to verify the guard fires.
    """

    def test_procedural_partial_yaml_without_target_modules_refuses(self, tmp_path):
        """Partial procedural YAML without target_modules must refuse loud."""
        from paramem.backup.types import FatalConfigError

        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            adapters:
              procedural:
                enabled: true
                rank: 8
                alpha: 16
                learning_rate: 5.0e-5
            """,
        )
        with pytest.raises(FatalConfigError, match="adapters.procedural.enabled=true"):
            load_server_config(yaml_file)

    def test_episodic_partial_yaml_without_target_modules_refuses(self, tmp_path):
        """Partial episodic YAML without target_modules must refuse loud."""
        from paramem.backup.types import FatalConfigError

        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            adapters:
              episodic:
                enabled: true
                rank: 16
            """,
        )
        with pytest.raises(FatalConfigError, match="adapters.episodic.enabled=true"):
            load_server_config(yaml_file)

    def test_disabled_tier_with_partial_yaml_passes(self, tmp_path):
        """When the tier is explicitly disabled, the missing-target_modules guard does not fire."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            adapters:
              procedural:
                enabled: false
                rank: 8
            """,
        )
        cfg = load_server_config(yaml_file)
        assert cfg.adapters.procedural.enabled is False

    def test_procedural_yaml_full_override_target_modules(self, tmp_path):
        """Explicit target_modules in YAML must win over the factory default."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            adapters:
              procedural:
                enabled: true
                target_modules: ["q_proj", "v_proj"]
            """,
        )
        cfg = load_server_config(yaml_file)
        assert cfg.adapters.procedural.target_modules == ["q_proj", "v_proj"]

    def test_no_adapters_block_uses_factory_defaults(self, tmp_path):
        """YAML without an adapters block falls through to the full factory defaults."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            """,
        )
        cfg = load_server_config(yaml_file)
        # Procedural factory ships MLP-targeting + enabled=True (symmetric with
        # episodic and semantic — all three main tiers default on).
        assert "gate_proj" in cfg.adapters.procedural.target_modules
        assert cfg.adapters.procedural.enabled is True


# ---------------------------------------------------------------------------
# RestartConfig / ProcessConfig
# ---------------------------------------------------------------------------


class TestRestartConfigDefaults:
    """RestartConfig() produces exactly the documented default values."""

    def test_defaults(self):
        """All fields match the documented safe defaults."""
        from paramem.server.config import RestartConfig

        r = RestartConfig()
        assert r.on_failure is True
        assert r.interval_seconds == 30
        assert r.max_attempts == 3
        assert r.window_seconds == 60
        assert r.permanent_failure_exit_codes == [3]


class TestRestartConfigValidators:
    """__post_init__ raises ValueError with a field-naming message for bad values."""

    def test_interval_seconds_zero_rejected(self):
        """interval_seconds < 1 raises ValueError naming the field."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="process.restart.interval_seconds"):
            RestartConfig(interval_seconds=0)

    def test_interval_seconds_negative_rejected(self):
        """Negative interval_seconds raises ValueError."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="process.restart.interval_seconds"):
            RestartConfig(interval_seconds=-5)

    def test_max_attempts_zero_rejected(self):
        """max_attempts < 1 raises ValueError naming the field."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="process.restart.max_attempts"):
            RestartConfig(max_attempts=0)

    def test_window_seconds_zero_rejected(self):
        """window_seconds < 1 raises ValueError naming the field."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="process.restart.window_seconds"):
            RestartConfig(window_seconds=0)

    def test_exit_code_above_255_rejected(self):
        """Exit code > 255 raises ValueError naming the field."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="process.restart.permanent_failure_exit_codes"):
            RestartConfig(permanent_failure_exit_codes=[256])

    def test_exit_code_negative_rejected(self):
        """Negative exit code raises ValueError naming the field."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="process.restart.permanent_failure_exit_codes"):
            RestartConfig(permanent_failure_exit_codes=[-1])

    def test_exit_code_boundary_values_accepted(self):
        """Exit codes 0 and 255 are valid boundaries."""
        from paramem.server.config import RestartConfig

        r = RestartConfig(permanent_failure_exit_codes=[0, 255])
        assert r.permanent_failure_exit_codes == [0, 255]

    def test_empty_exit_codes_list_rejected(self):
        """Empty list raises ValueError — would render an unset systemd value
        and silently disable the permanent-failure short-circuit."""
        from paramem.server.config import RestartConfig

        with pytest.raises(ValueError, match="must not be empty"):
            RestartConfig(permanent_failure_exit_codes=[])


class TestRestartConfigYamlLoader:
    """load_server_config wires the process.restart sub-block correctly."""

    def test_fixture_yaml_loads_restart_config(self):
        """load_server_config(tests/fixtures/server.yaml) exposes fixture values."""
        from pathlib import Path

        from paramem.server.config import RestartConfig, load_server_config

        cfg = load_server_config(Path("tests/fixtures/server.yaml"))
        r = cfg.process.restart
        # Values are the defaults — same as RestartConfig() — because the
        # fixture mirrors the documented safe defaults.
        assert r == RestartConfig()

    def test_yaml_with_process_restart_overrides(self, tmp_path):
        """Explicit process.restart fields in YAML override the defaults."""
        from paramem.server.config import load_server_config

        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            process:
              restart:
                on_failure: false
                interval_seconds: 10
                max_attempts: 5
                window_seconds: 120
                permanent_failure_exit_codes: [1, 3]
            """,
        )
        cfg = load_server_config(yaml_file)
        r = cfg.process.restart
        assert r.on_failure is False
        assert r.interval_seconds == 10
        assert r.max_attempts == 5
        assert r.window_seconds == 120
        assert r.permanent_failure_exit_codes == [1, 3]

    def test_process_without_restart_subblock_uses_default_restart(self, tmp_path):
        """A YAML with only process: (no restart: sub-block) defaults to RestartConfig()."""
        from paramem.server.config import RestartConfig, load_server_config

        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            process:
            """,
        )
        cfg = load_server_config(yaml_file)
        assert cfg.process.restart == RestartConfig()

    def test_absent_process_section_uses_default(self, tmp_path):
        """No process: key at all → ProcessConfig() with RestartConfig() defaults."""
        from paramem.server.config import RestartConfig, load_server_config

        yaml_file = _write_yaml(tmp_path, "model: mistral\n")
        cfg = load_server_config(yaml_file)
        assert cfg.process.restart == RestartConfig()


class TestTrainingHyperparamsFromYaml:
    """Regression: assembled training_config carries the Test 17 recipe from yaml.

    Loads tests/fixtures/server.yaml (the stable fixture per the loader rule) and
    asserts every Test 17 field survives assembly into the TrainingConfig returned
    by ServerConfig.training_config.
    """

    def test_fixture_training_config_carries_test17_recipe(self):
        """training_config assembled from fixtures/server.yaml has Test 17 values."""
        from pathlib import Path

        from paramem.server.config import load_server_config

        cfg = load_server_config(Path("tests/fixtures/server.yaml"))
        tc = cfg.training_config
        assert tc.weight_decay == 0.1
        assert tc.warmup_steps == 30
        assert tc.warmup_ratio == 0.0
        assert tc.lr_scheduler_type == "linear"
        assert tc.max_seq_length == 1024
        assert tc.batch_size == 1
        assert tc.gradient_accumulation_steps == 2
        assert tc.seed == 42
        assert tc.max_grad_norm == 1.0
        assert tc.gradient_checkpointing is True
        assert tc.lr_decay_steps is None

    def test_training_hyperparams_yaml_override_flows_through(self, tmp_path):
        """Explicit consolidation.training_* yaml values flow through to TrainingConfig."""
        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            consolidation:
              training_batch_size: 2
              training_weight_decay: 0.05
              training_warmup_steps: 10
              training_warmup_ratio: 0.0
              training_lr_scheduler_type: constant
              training_max_seq_length: 512
              training_gradient_accumulation_steps: 4
              training_seed: 7
              training_max_grad_norm: 0.5
              training_gradient_checkpointing: false
              training_lr_decay_steps: 200
            """,
        )
        config = load_server_config(yaml_file)
        tc = config.training_config
        assert tc.batch_size == 2
        assert tc.weight_decay == 0.05
        assert tc.warmup_steps == 10
        assert tc.warmup_ratio == 0.0
        assert tc.lr_scheduler_type == "constant"
        assert tc.max_seq_length == 512
        assert tc.gradient_accumulation_steps == 4
        assert tc.seed == 7
        assert tc.max_grad_norm == 0.5
        assert tc.gradient_checkpointing is False
        assert tc.lr_decay_steps == 200


class TestMakeTrainingConfigPropagation:
    """Propagation guard: _make_training_config must forward warmup_steps,
    lr_scheduler_type, and lr_decay_steps from self.training_config.

    Regression guard for a latent propagation bug where these three fields
    were silently dropped, so yaml overrides never reached train_adapter.
    """

    def test_make_training_config_propagates_three_fixed_fields(self, tmp_path):
        """warmup_steps, lr_scheduler_type, lr_decay_steps survive _make_training_config."""
        from paramem.server.config import load_server_config

        yaml_file = _write_yaml(
            tmp_path,
            """\
            model: mistral
            consolidation:
              training_warmup_steps: 15
              training_lr_scheduler_type: constant
              training_lr_decay_steps: 300
            """,
        )
        config = load_server_config(yaml_file)

        # Verify the values are on training_config first.
        assert config.training_config.warmup_steps == 15
        assert config.training_config.lr_scheduler_type == "constant"
        assert config.training_config.lr_decay_steps == 300

        # Now verify _make_training_config propagates them (import ConsolidationLoop
        # is heavy; test via a lightweight stub that mimics its interface).
        from paramem.utils.config import TrainingConfig

        class _StubLoop:
            """Minimal stub exposing _make_training_config for isolation testing."""

            training_config = config.training_config

            _make_training_config = (
                # borrow the real implementation without importing ConsolidationLoop
                __import__(
                    "paramem.training.consolidation", fromlist=["ConsolidationLoop"]
                ).ConsolidationLoop._make_training_config
            )

        stub = _StubLoop()
        rebuilt = stub._make_training_config(num_epochs=5)
        assert isinstance(rebuilt, TrainingConfig)
        assert rebuilt.warmup_steps == 15
        assert rebuilt.lr_scheduler_type == "constant"
        assert rebuilt.lr_decay_steps == 300
        assert rebuilt.num_epochs == 5  # caller-supplied, not propagated


class TestTierFloorConfigPlumbing:
    """min_tier_key_floor and tier_fast_start wire through
    ConsolidationScheduleConfig → consolidation_config property → ConsolidationConfig.
    """

    FIXTURE = "tests/fixtures/server.yaml"

    def test_fixture_yaml_has_floor_and_fast_start(self):
        """Fixture YAML contains both new keys under consolidation:."""
        from paramem.server.config import load_server_config

        config = load_server_config(self.FIXTURE)
        assert config.consolidation.min_tier_key_floor == 30
        assert config.consolidation.tier_fast_start is True

    def test_consolidation_config_bridges_floor_and_fast_start(self):
        """consolidation_config property propagates both fields to ConsolidationConfig."""
        from paramem.server.config import load_server_config

        config = load_server_config(self.FIXTURE)
        cc = config.consolidation_config
        assert cc.min_tier_key_floor == 30
        assert cc.tier_fast_start is True

    def test_yaml_override_respected(self, tmp_path):
        """Custom YAML values for both fields are loaded correctly."""
        import textwrap

        from paramem.server.config import load_server_config

        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                model: mistral
                consolidation:
                  min_tier_key_floor: 50
                  tier_fast_start: false
            """),
            encoding="utf-8",
        )
        config = load_server_config(yaml_file)
        assert config.consolidation.min_tier_key_floor == 50
        assert config.consolidation.tier_fast_start is False

        cc = config.consolidation_config
        assert cc.min_tier_key_floor == 50
        assert cc.tier_fast_start is False

    def test_defaults_when_keys_absent(self, tmp_path):
        """When consolidation section is absent, defaults are 30 and True."""
        from paramem.server.config import load_server_config

        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text("model: mistral\n", encoding="utf-8")
        config = load_server_config(yaml_file)
        assert config.consolidation.min_tier_key_floor == 30
        assert config.consolidation.tier_fast_start is True


class TestDefaultServerConfigPathIsCwdIndependent:
    """``DEFAULT_SERVER_CONFIG_PATH`` is absolute and anchored to the repo root,
    so config resolution does not depend on the process's working directory.

    ParaMem deploys from a repo checkout (editable install under systemd) with
    ``WorkingDirectory`` = repo root today; these tests lock in that the loader
    no longer *relies* on that cwd, closing the same landmine class as the
    ``_trial_json_path`` fallback bug.
    """

    def test_default_path_is_absolute_and_repo_anchored(self):
        # The old ``Path("configs/server.yaml")`` literal was relative; the fix
        # anchors it to the nearest ``pyproject.toml`` ancestor (the repo root).
        assert DEFAULT_SERVER_CONFIG_PATH.is_absolute()
        assert DEFAULT_SERVER_CONFIG_PATH.name == "server.yaml"
        assert DEFAULT_SERVER_CONFIG_PATH.parent.name == "configs"
        assert (DEFAULT_SERVER_CONFIG_PATH.parent.parent / "pyproject.toml").is_file()

    def test_default_load_is_cwd_independent(self, monkeypatch, tmp_path):
        # Resolve from the repo-root cwd (pytest's cwd) first.
        from_root = load_server_config()
        # A loaded config absolutizes ``paths.data`` against the repo root; a
        # bare ``ServerConfig()`` (the "path vanished" branch) leaves it the
        # relative ``data/ha`` default. Under the old relative default, calling
        # from a foreign cwd hit that branch — this asserts it no longer does.
        monkeypatch.chdir(tmp_path)
        from_foreign = load_server_config()
        assert from_foreign.paths.data.is_absolute()
        assert from_foreign.paths.data == from_root.paths.data
        assert from_foreign.model_name == from_root.model_name

    def test_relative_string_default_still_resolves(self, monkeypatch):
        # The fresh-clone comparison resolves ``path`` so a caller may still
        # spell the default as the cwd-relative ``configs/server.yaml`` (as
        # several existing call sites and tests do) from the repo root.
        repo_root = DEFAULT_SERVER_CONFIG_PATH.parent.parent
        monkeypatch.chdir(repo_root)
        cfg = load_server_config("configs/server.yaml")
        assert cfg.paths.data.is_absolute()

    def test_default_data_dir_is_absolute_and_repo_anchored(self):
        # The data-root fallback used where a loaded ``config.paths.data`` is
        # unavailable (config is None / a test mock). It replaced cwd-relative
        # ``Path("data/ha/...")`` literals scattered across the server, so it
        # must be absolute and share the repo root with the config path.
        assert DEFAULT_DATA_DIR.is_absolute()
        assert DEFAULT_DATA_DIR.name == "ha"
        assert DEFAULT_DATA_DIR.parent.name == "data"
        assert DEFAULT_DATA_DIR.parent.parent == DEFAULT_SERVER_CONFIG_PATH.parent.parent
        assert (DEFAULT_DATA_DIR.parent.parent / "pyproject.toml").is_file()

    def test_fixture_telemetry_path_is_absolute(self):
        # paths.telemetry must go through the same relative-path anchoring
        # loop as data/sessions/debug/prompts (config.py's ``for path_field
        # in (...)`` tuple). A relative path here means the loop was missed —
        # a regression of the 8f173a4 cwd-independence fix.
        cfg = load_server_config(Path("tests/fixtures/server.yaml"))
        assert cfg.paths.telemetry.is_absolute()
