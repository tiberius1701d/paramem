"""Tests for paramem.backup.runner — run_scheduled_backup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.backup.runner import ScheduledBackupResult, run_scheduled_backup
from paramem.server.config import (
    PathsConfig,
    RetentionConfig,
    RetentionTierConfig,
    SecurityConfig,
    ServerAdaptersConfig,
    ServerBackupsConfig,
    ServerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_config(
    tmp_path: Path,
    schedule: str = "daily 04:00",
    artifacts: list[str] | None = None,
    max_total_disk_gb: float = 20.0,
) -> ServerConfig:
    """Build a minimal ServerConfig pointing at tmp_path (no adapter dirs)."""
    if artifacts is None:
        artifacts = ["config", "graph", "registry"]
    config = ServerConfig.__new__(ServerConfig)
    paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.paths = paths
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule=schedule,
            artifacts=artifacts,
            max_total_disk_gb=max_total_disk_gb,
            retention=RetentionConfig(
                daily=RetentionTierConfig(keep=7),
                manual=RetentionTierConfig(keep="unlimited", max_disk_gb=5.0),
            ),
        )
    )
    return config


def _make_bundle_server_config(
    tmp_path: Path,
    schedule: str = "daily 04:00",
    max_total_disk_gb: float = 20.0,
) -> ServerConfig:
    """Build a ServerConfig with adapters enabled, pointing at tmp_path.

    Uses a real ``ServerAdaptersConfig`` so the runner's bundle branch can
    iterate ``config.adapters.<tier>.enabled``.  Only episodic is enabled
    (the primary recall tier that the bundle writer requires).
    """
    from paramem.server.config import ServerAdapterConfig

    config = ServerConfig.__new__(ServerConfig)
    paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.paths = paths
    # Only episodic enabled — the minimum for a valid bundle.
    config.adapters = ServerAdaptersConfig(
        episodic=ServerAdapterConfig(enabled=True),
        semantic=ServerAdapterConfig(enabled=False),
        procedural=ServerAdapterConfig(enabled=False),
    )
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule=schedule,
            artifacts=["snapshot_bundle"],
            max_total_disk_gb=max_total_disk_gb,
            retention=RetentionConfig(
                daily=RetentionTierConfig(keep=7),
                manual=RetentionTierConfig(keep="unlimited", max_disk_gb=5.0),
            ),
        )
    )
    return config


def _seed_episodic_adapter(config: ServerConfig, indexed_key_content: bytes) -> str:
    """Create a minimal episodic adapter slot matching indexed_key_content hash.

    Returns the registry_sha256 so callers can verify the bundle manifest.
    """
    registry_sha256 = hashlib.sha256(indexed_key_content).hexdigest()
    episodic_dir = config.paths.adapters / "episodic"
    episodic_dir.mkdir(parents=True, exist_ok=True)

    # Write indexed_key_registry.json for per-tier hash resolution.
    (episodic_dir / "indexed_key_registry.json").write_bytes(indexed_key_content)

    # Create the live adapter slot stamped with the matching registry_sha256.
    slot_dir = episodic_dir / "20260520-123456"
    slot_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 4,
        "name": "episodic",
        "trained_at": "2026-05-20T12:00:00Z",
        "window_stamp": "",
        "base_model": {
            "repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "sha": "c170c708abc",
            "hash": "sha256:f4b6abc",
        },
        "tokenizer": {
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
            "vocab_size": 32000,
            "merges_hash": "d" * 64,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
        "registry_sha256": registry_sha256,
        "key_count": 10,
        "synthesized": False,
    }
    (slot_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (slot_dir / "adapter_model.safetensors").write_bytes(b"fake_weights")
    (slot_dir / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
    return registry_sha256


def _mock_loop(graph_bytes: bytes = b'{"nodes": []}') -> MagicMock:
    """Return a mock ConsolidationLoop with a merger that saves bytes."""
    loop = MagicMock()
    loop.merger = MagicMock()
    loop.merger.save_bytes.return_value = graph_bytes
    return loop


def _run(
    tmp_path: Path,
    schedule: str = "daily 04:00",
    artifacts: list[str] | None = None,
    loop=None,
    max_total_disk_gb: float = 20.0,
    config_content: bytes | None = b"model: mistral\n",
    write_registry: bool = True,
) -> tuple[ScheduledBackupResult, ServerConfig, Path, Path]:
    """Run the backup runner and return (result, config, state_dir, backups_root)."""
    config = _make_server_config(
        tmp_path,
        schedule=schedule,
        artifacts=artifacts,
        max_total_disk_gb=max_total_disk_gb,
    )
    # Setup directories.
    state_dir = (config.paths.data / "state").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    backups_root = (config.paths.data / "backups").resolve()
    backups_root.mkdir(parents=True, exist_ok=True)
    config.paths.data.mkdir(parents=True, exist_ok=True)

    # Write live config file.
    live_config_path = tmp_path / "server.yaml"
    if config_content is not None:
        live_config_path.write_bytes(config_content)

    # Write registry if requested.
    key_metadata_path = config.paths.key_metadata
    key_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if write_registry:
        key_metadata_path.write_text('{"keys": {}}', encoding="utf-8")

    result = run_scheduled_backup(
        server_config=config,
        loop=loop,
        state_dir=state_dir,
        backups_root=backups_root,
        live_config_path=live_config_path,
    )
    return result, config, state_dir, backups_root


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestRunnerSuccess:
    def test_runner_writes_three_artifacts_on_success(self, tmp_path):
        """Default config + mock loop → 3 slots written, prune ran."""
        loop = _mock_loop()
        result, _, _, backups_root = _run(tmp_path, loop=loop)
        assert result.success
        assert set(result.written_slots.keys()) == {"config", "graph", "registry"}
        # Each slot dir exists.
        for name, path_str in result.written_slots.items():
            assert Path(path_str).exists(), f"{name} slot not on disk"

    def test_runner_uses_artifacts_subset(self, tmp_path):
        """artifacts=["config"] → only config written."""
        result, _, _, _ = _run(tmp_path, artifacts=["config"])
        assert result.success
        assert set(result.written_slots.keys()) == {"config"}

    def test_runner_writes_meta_tier_field(self, tmp_path):
        """Inspect written slot's meta.json — tier="daily"."""
        loop = _mock_loop()
        result, _, _, backups_root = _run(tmp_path, loop=loop)
        slot_path = Path(result.written_slots["config"])
        meta_files = list(slot_path.glob("*.meta.json"))
        assert meta_files
        meta = json.loads(meta_files[0].read_text())
        assert meta["tier"] == "daily"

    def test_runner_passes_label_to_meta(self, tmp_path):
        """label='test-label' → meta.label=='test-label'."""
        config = _make_server_config(tmp_path)
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = (config.paths.data / "backups").resolve()
        backups_root.mkdir(parents=True, exist_ok=True)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.write_text("{}", encoding="utf-8")
        result = run_scheduled_backup(
            server_config=config,
            loop=None,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=live_config,
            label="test-label",
        )
        slot_path = Path(result.written_slots["config"])
        meta_files = list(slot_path.glob("*.meta.json"))
        assert meta_files
        meta = json.loads(meta_files[0].read_text())
        assert meta["label"] == "test-label"


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------


class TestRunnerSkipPaths:
    def test_runner_skips_graph_when_loop_none(self, tmp_path):
        """loop=None → graph in skipped_artifacts; config + registry written."""
        result, _, _, _ = _run(tmp_path, loop=None)
        assert result.success
        assert "graph" in {a for a, _ in result.skipped_artifacts}
        assert "config" in result.written_slots
        assert "registry" in result.written_slots
        skip_reasons = {a: r for a, r in result.skipped_artifacts}
        assert "unavailable" in skip_reasons["graph"].lower()

    def test_runner_skips_registry_when_missing(self, tmp_path):
        """key_metadata.json absent → registry skipped with reason."""
        result, _, _, _ = _run(tmp_path, loop=None, write_registry=False)
        assert result.success
        assert "registry" in {a for a, _ in result.skipped_artifacts}
        skip_reasons = {a: r for a, r in result.skipped_artifacts}
        assert "registry" in skip_reasons["registry"].lower()

    def test_runner_schedule_off_returns_noop(self, tmp_path):
        """schedule="off" → success=True, written_slots empty."""
        result, _, _, _ = _run(tmp_path, schedule="off")
        assert result.success
        assert result.written_slots == {}
        assert result.skipped_artifacts == []
        assert result.prune_result_summary is None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestRunnerFailurePaths:
    def test_runner_aborts_after_first_failure(self, tmp_path):
        """Force backup_write to raise on config → registry skipped with 'aborted'."""
        loop = _mock_loop()

        def _raise(*args, **kwargs):
            raise OSError("disk full")

        with patch("paramem.backup.backup.write", side_effect=_raise):
            result, _, _, _ = _run(tmp_path, loop=loop)

        assert not result.success
        assert result.error is not None
        aborted = [a for a, r in result.skipped_artifacts if "aborted" in r]
        assert len(aborted) > 0

    def test_runner_disk_pressure_refuses_writes(self, tmp_path):
        """max_total_disk_gb=0.000001 (1 KB) → success=False, error=disk_pressure."""
        loop = _mock_loop()
        # Create a 10 KB file in the backups dir so usage > cap.
        config = _make_server_config(tmp_path, max_total_disk_gb=0.000001)
        backups_root = config.paths.data / "backups" / "config" / "20260401-040000"
        backups_root.mkdir(parents=True, exist_ok=True)
        (backups_root / "data.bin").write_bytes(b"x" * 10240)
        (backups_root / "data.meta.json").write_text(
            json.dumps({"tier": "daily", "schema_version": 1}), encoding="utf-8"
        )

        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.write_text("{}")

        br = (config.paths.data / "backups").resolve()
        result = run_scheduled_backup(
            server_config=config,
            loop=loop,
            state_dir=state_dir,
            backups_root=br,
            live_config_path=live_config,
        )
        assert not result.success
        assert "disk_pressure" in (result.error or "")
        assert result.written_slots == {}

    def test_runner_prune_failure_does_not_fail_backup(self, tmp_path):
        """prune() raises → success still True, prune_result_summary is None."""
        loop = _mock_loop()

        with patch("paramem.backup.retention.prune", side_effect=RuntimeError("prune exploded")):
            result, _, _, _ = _run(tmp_path, loop=loop)

        assert result.success
        assert result.prune_result_summary is None


# ---------------------------------------------------------------------------
# Prune called
# ---------------------------------------------------------------------------


class TestRunnerCallsPrune:
    def test_runner_calls_prune_after_success(self, tmp_path):
        """Mock prune; verify called once with the right kwargs."""
        loop = _mock_loop()

        with patch("paramem.backup.retention.prune") as mock_prune:
            mock_prune.return_value = MagicMock(
                deleted=[],
                preserved_immune=[],
                preserved_migration_window=[],
                invalid_slots=[],
                disk_usage_after=MagicMock(total_bytes=0, cap_bytes=0),
            )
            result, _, _, backups_root = _run(tmp_path, loop=loop)

        assert result.success
        mock_prune.assert_called_once()
        call_kwargs = mock_prune.call_args.kwargs
        assert call_kwargs["backups_root"] == backups_root
        assert call_kwargs["dry_run"] is False


# ---------------------------------------------------------------------------
# Cleanup 2 — keep=0 short-circuit skips emission
# ---------------------------------------------------------------------------


class TestRunnerSkipsEmissionWhenKeepIsZero:
    def test_runner_skips_emission_when_target_tier_keep_is_zero(self, tmp_path):
        """daily.keep=0 → runner returns success without writing any slots.

        Verifies three properties:
        - success=True and written_slots={} (no write performed).
        - All configured artifacts appear in skipped_artifacts with reason
          indicating the tier keep=0 short-circuit.
        - No slot directories are created under backups_root.
        """
        config = _make_server_config(tmp_path)
        # Set daily.keep to 0 directly — RetentionTierConfig is a plain mutable dataclass.
        config.security.backups.retention.daily.keep = 0

        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = (config.paths.data / "backups").resolve()
        backups_root.mkdir(parents=True, exist_ok=True)
        config.paths.data.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.write_text('{"keys": {}}', encoding="utf-8")

        loop = _mock_loop()
        result = run_scheduled_backup(
            server_config=config,
            loop=loop,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=live_config,
            tier="daily",
        )

        # Must succeed without writing.
        assert result.success
        assert result.written_slots == {}
        assert result.error is None
        assert result.prune_result_summary is None

        # Every configured artifact must appear in skipped_artifacts with the
        # keep=0 reason, not a false failure marker.
        skip_reasons = {a: r for a, r in result.skipped_artifacts}
        for artifact in config.security.backups.artifacts:
            assert artifact in skip_reasons, f"{artifact} not in skipped_artifacts"
            assert "keep=0" in skip_reasons[artifact]

        # No slot directories created under backups_root.
        artifact_dirs = (
            [d for d in backups_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if backups_root.exists()
            else []
        )
        assert artifact_dirs == [], f"Unexpected dirs created: {artifact_dirs}"

        # State file must reflect no failure when the caller propagates the result.
        from paramem.backup.state import read_backup_state, update_backup_state

        update_backup_state(state_dir, result)
        state = read_backup_state(state_dir)
        assert state is not None
        assert state.last_failure_at is None, (
            "keep=0 no-op must not write a failure timestamp to the state file"
        )


# ---------------------------------------------------------------------------
# Bundle path (artifact = "snapshot_bundle")
# ---------------------------------------------------------------------------


class TestRunnerBundlePath:
    """Tests for the snapshot_bundle artifact branch in BackupRunner."""

    def _setup(self, tmp_path: Path):
        """Set up dirs, config, adapter fixture, and live config."""
        config = _make_bundle_server_config(tmp_path)
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = (config.paths.data / "backups").resolve()
        backups_root.mkdir(parents=True, exist_ok=True)
        config.paths.data.mkdir(parents=True, exist_ok=True)

        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        # key_metadata.json (global registry — does not drive slot resolution for
        # the bundle, but write_bundle captures it as "registry/key_metadata.json").
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.write_text('{"keys": {}}', encoding="utf-8")

        # Seed episodic adapter slot so write_bundle can find the live slot.
        indexed_key_content = b'{"keys": {"k1": "v1"}}'
        registry_sha256 = _seed_episodic_adapter(config, indexed_key_content)

        return config, state_dir, backups_root, live_config, registry_sha256

    def test_bundle_produces_snapshot_bundle_slot(self, tmp_path: Path):
        """snapshot_bundle artifact → written_slots == {"snapshot_bundle": <path>}."""
        config, state_dir, backups_root, live_config, _ = self._setup(tmp_path)

        result = run_scheduled_backup(
            server_config=config,
            loop=None,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=live_config,
        )

        assert result.success, f"Bundle backup failed: {result.error}"
        assert set(result.written_slots.keys()) == {"snapshot_bundle"}
        slot_path = Path(result.written_slots["snapshot_bundle"])
        assert slot_path.exists(), f"Bundle slot directory not found: {slot_path}"
        assert (slot_path / "bundle.meta.json").exists(), "bundle.meta.json missing"

    def test_bundle_slot_in_snapshot_subdir(self, tmp_path: Path):
        """Bundle slots land under backups_root/snapshot/, not per-artifact dirs."""
        config, state_dir, backups_root, live_config, _ = self._setup(tmp_path)

        result = run_scheduled_backup(
            server_config=config,
            loop=None,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=live_config,
        )

        assert result.success
        slot_path = Path(result.written_slots["snapshot_bundle"])
        # The slot must be inside backups_root/snapshot/<ts>/.
        assert slot_path.parent == backups_root / "snapshot"

    def test_bundle_contains_episodic_adapter(self, tmp_path: Path):
        """Bundle manifest records the episodic adapter entry."""
        config, state_dir, backups_root, live_config, _ = self._setup(tmp_path)

        result = run_scheduled_backup(
            server_config=config,
            loop=None,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=live_config,
        )

        assert result.success
        slot_path = Path(result.written_slots["snapshot_bundle"])
        manifest = json.loads((slot_path / "bundle.meta.json").read_text())
        assert "episodic" in manifest["adapters"], "Episodic adapter missing from manifest"

    def test_bundle_prune_ran(self, tmp_path: Path):
        """Prune runs after a successful bundle write."""
        config, state_dir, backups_root, live_config, _ = self._setup(tmp_path)

        with patch("paramem.backup.retention.prune") as mock_prune:
            mock_prune.return_value = MagicMock(
                deleted=[],
                preserved_immune=[],
                preserved_migration_window=[],
                invalid_slots=[],
                disk_usage_after=MagicMock(total_bytes=0, cap_bytes=0),
            )
            result = run_scheduled_backup(
                server_config=config,
                loop=None,
                state_dir=state_dir,
                backups_root=backups_root,
                live_config_path=live_config,
            )

        assert result.success
        mock_prune.assert_called_once()

    def test_bundle_write_error_returns_failure(self, tmp_path: Path):
        """write_bundle BackupError → success=False, error captured."""
        config, state_dir, backups_root, live_config, _ = self._setup(tmp_path)

        from paramem.backup.types import BackupError

        def _raise(*args, **kwargs):
            raise BackupError("episodic missing")

        with patch("paramem.backup.backup.write_bundle", side_effect=_raise):
            result = run_scheduled_backup(
                server_config=config,
                loop=None,
                state_dir=state_dir,
                backups_root=backups_root,
                live_config_path=live_config,
            )

        assert not result.success
        assert result.error is not None
        assert "episodic missing" in result.error

    def test_bundle_schedule_off_noop(self, tmp_path: Path):
        """schedule=off → success=True, no bundle written."""
        config = _make_bundle_server_config(tmp_path, schedule="off")
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = (config.paths.data / "backups").resolve()
        backups_root.mkdir(parents=True, exist_ok=True)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")

        result = run_scheduled_backup(
            server_config=config,
            loop=None,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=live_config,
        )

        assert result.success
        assert result.written_slots == {}

    def test_bundle_disk_pressure_refuses_write(self, tmp_path: Path):
        """Disk over cap → success=False, snapshot_bundle in skipped_artifacts."""
        config = _make_bundle_server_config(tmp_path, max_total_disk_gb=0.000001)
        # Put a file over the cap in the backups dir.
        bloat_dir = config.paths.data / "backups" / "snapshot" / "20260401-040000"
        bloat_dir.mkdir(parents=True, exist_ok=True)
        (bloat_dir / "data.bin").write_bytes(b"x" * 10240)

        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        live_config = tmp_path / "server.yaml"
        live_config.write_bytes(b"model: mistral\n")
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata.write_text("{}")

        result = run_scheduled_backup(
            server_config=config,
            loop=None,
            state_dir=state_dir,
            backups_root=(config.paths.data / "backups").resolve(),
            live_config_path=live_config,
        )

        assert not result.success
        assert "disk_pressure" in (result.error or "")
