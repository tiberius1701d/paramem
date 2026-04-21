"""Unit tests for scripts/migrate/outputs_to_slot_dirs.py.

Tests cover:
- Flat-layout adapter → slot layout (weights relocated, meta.json written).
- meta.json has synthesized=True.
- Idempotent rerun is a no-op.
- dry-run writes nothing and skips liveness check.
- Sibling registry SHA-256 is embedded.
- Missing sibling registry → UNKNOWN.
- Name heuristic: bare dir name and adapter/parent.name branches.
- Synthesized manifest round-trips through read_manifest with synthesized=True.
- --force bypasses liveness check (mock pgrep).
- Absent --force with alive PID → exit 1 + stderr message (mock pgrep).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure repo root is on sys.path so the migration module can be imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.migrate.outputs_to_slot_dirs import (  # noqa: E402
    _adapter_name_from_dir,
    _discover_old_layout_dirs,
    _is_old_layout,
    _reshape_dir,
    migrate,
)


def _make_flat_adapter(base: Path, name: str = "episodic") -> Path:
    """Create a minimal old-layout flat adapter directory.

    Args:
        base: Parent directory (e.g. ``tmp_path / "run1"``).
        name: Adapter directory name.

    Returns:
        Path to the newly created flat adapter directory.
    """
    adapter_dir = base / name
    adapter_dir.mkdir(parents=True)
    cfg = {
        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake weights")
    return adapter_dir


class TestIsOldLayout:
    """_is_old_layout correctly identifies old-layout directories."""

    def test_flat_with_both_files_is_old(self, tmp_path):
        d = _make_flat_adapter(tmp_path)
        assert _is_old_layout(d)

    def test_slot_name_pattern_is_skipped(self, tmp_path):
        slot = tmp_path / "20260420-120000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"x")
        assert not _is_old_layout(slot)

    def test_meta_json_present_is_skipped(self, tmp_path):
        d = _make_flat_adapter(tmp_path)
        (d / "meta.json").write_text("{}")
        assert not _is_old_layout(d)

    def test_missing_safetensors_is_not_old(self, tmp_path):
        d = tmp_path / "episodic"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        assert not _is_old_layout(d)

    def test_missing_config_is_not_old(self, tmp_path):
        d = tmp_path / "episodic"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(b"x")
        assert not _is_old_layout(d)


class TestDiscoverOldLayoutDirs:
    """_discover_old_layout_dirs finds all matching directories recursively."""

    def test_finds_nested_old_layouts(self, tmp_path):
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        _make_flat_adapter(run1, "episodic")
        _make_flat_adapter(run2, "semantic")
        found = _discover_old_layout_dirs(tmp_path)
        assert len(found) == 2
        names = {d.name for d in found}
        assert names == {"episodic", "semantic"}

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".pending" / "episodic"
        hidden.mkdir(parents=True)
        (hidden / "adapter_config.json").write_text("{}")
        (hidden / "adapter_model.safetensors").write_bytes(b"x")
        found = _discover_old_layout_dirs(tmp_path)
        assert found == []


class TestReshapeDir:
    """_reshape_dir correctly restructures a flat adapter directory."""

    def test_flat_to_slot_layout(self, tmp_path):
        """After reshape: weights in slot, meta.json present."""
        adapter_dir = _make_flat_adapter(tmp_path)
        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        # Original dir is gone (renamed).
        assert not adapter_dir.exists()
        # Slot dir must be a direct child of tmp_path.
        slots = [d for d in tmp_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) == 1
        slot = slots[0]
        assert (slot / "adapter_model.safetensors").exists()
        assert (slot / "adapter_config.json").exists()
        assert (slot / "meta.json").exists()

    def test_meta_json_has_synthesized_true(self, tmp_path):
        """Synthesized manifest must have synthesized=True."""
        from paramem.adapters.manifest import read_manifest

        adapter_dir = _make_flat_adapter(tmp_path)
        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        slots = [d for d in tmp_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        manifest = read_manifest(slots[0])
        assert manifest.synthesized is True

    def test_dry_run_writes_nothing(self, tmp_path):
        """dry_run=True must not modify the filesystem."""
        adapter_dir = _make_flat_adapter(tmp_path)
        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=True,
            verbose=False,
        )
        # Original dir unchanged.
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_model.safetensors").exists()
        # No new slot dirs.
        other_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d != adapter_dir]
        assert other_dirs == []

    def test_sibling_registry_sha256_embedded(self, tmp_path):
        """manifest.registry_sha256 matches sha256 of sibling registry."""
        import hashlib

        from paramem.adapters.manifest import read_manifest

        adapter_dir = _make_flat_adapter(tmp_path)
        registry_path = tmp_path / "indexed_key_registry.json"
        registry_path.write_text('{"active_keys": ["graph1"]}')
        expected = hashlib.sha256(registry_path.read_bytes()).hexdigest()

        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        slots = [d for d in tmp_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        manifest = read_manifest(slots[0])
        assert manifest.registry_sha256 == expected

    def test_missing_registry_gives_unknown(self, tmp_path):
        """manifest.registry_sha256 == UNKNOWN when no registry file exists."""
        from paramem.adapters.manifest import UNKNOWN, read_manifest

        adapter_dir = _make_flat_adapter(tmp_path)
        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        slots = [d for d in tmp_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        manifest = read_manifest(slots[0])
        assert manifest.registry_sha256 == UNKNOWN

    def test_keyed_pairs_sha256_embedded(self, tmp_path):
        """manifest.keyed_pairs_sha256 matches sha256 of keyed_pairs.json."""
        import hashlib

        from paramem.adapters.manifest import read_manifest

        adapter_dir = _make_flat_adapter(tmp_path)
        kp_data = [{"key": "graph1", "question": "Q?", "answer": "A."}]
        (adapter_dir / "keyed_pairs.json").write_text(json.dumps(kp_data))
        expected = hashlib.sha256((adapter_dir / "keyed_pairs.json").read_bytes()).hexdigest()

        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        # After rename, keyed_pairs.json is inside the slot.
        slots = [d for d in tmp_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        manifest = read_manifest(slots[0])
        assert manifest.keyed_pairs_sha256 == expected

    def test_idempotent_rerun_is_noop(self, tmp_path):
        """A second pass after migration must skip the already-migrated slot."""
        adapter_dir = _make_flat_adapter(tmp_path)
        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        # Discover again — should find nothing new.
        found = _discover_old_layout_dirs(tmp_path)
        assert found == [], f"Expected no old-layout dirs after migration, got: {found}"


class TestNameHeuristic:
    """_adapter_name_from_dir returns the correct adapter name."""

    def test_plain_dir_name(self, tmp_path):
        d = tmp_path / "episodic"
        d.mkdir()
        assert _adapter_name_from_dir(d, name_from_config=False) == "episodic"

    def test_adapter_dir_name_uses_parent(self, tmp_path):
        """When dir.name == 'adapter', return parent.name."""
        parent = tmp_path / "episodic"
        parent.mkdir()
        d = parent / "adapter"
        d.mkdir()
        assert _adapter_name_from_dir(d, name_from_config=False) == "episodic"

    def test_name_from_config_reads_adapter_name(self, tmp_path):
        """With name_from_config=True, read from adapter_config.json."""
        d = tmp_path / "ep"
        d.mkdir()
        cfg = {"adapter_name": "episodic_main", "r": 4}
        (d / "adapter_config.json").write_text(json.dumps(cfg))
        assert _adapter_name_from_dir(d, name_from_config=True) == "episodic_main"

    def test_name_from_config_falls_back_to_dir_name(self, tmp_path):
        """With name_from_config=True and no adapter_name, fall back to dir name."""
        d = tmp_path / "episodic"
        d.mkdir()
        (d / "adapter_config.json").write_text('{"r": 4}')
        assert _adapter_name_from_dir(d, name_from_config=True) == "episodic"


class TestSynthesizedManifestRoundtrip:
    """Synthesized manifest round-trips through read_manifest preserving synthesized=True."""

    def test_round_trip(self, tmp_path):
        from paramem.adapters.manifest import read_manifest

        adapter_dir = _make_flat_adapter(tmp_path)
        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        slots = [d for d in tmp_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots
        m = read_manifest(slots[0])
        assert m.synthesized is True
        assert m.name == "episodic"


class TestMigrateFunction:
    """migrate() top-level function handles liveness check and dry-run correctly."""

    def test_dry_run_skips_liveness_check_and_writes_nothing(self, tmp_path):
        """dry-run must not call pgrep and must not modify the filesystem."""
        adapter_dir = _make_flat_adapter(tmp_path)
        pgrep_called = []

        def _fake_pgrep(patterns):
            pgrep_called.append(patterns)
            return []

        with patch("scripts.migrate.outputs_to_slot_dirs._pgrep_alive", side_effect=_fake_pgrep):
            rc = migrate(tmp_path, dry_run=True)

        assert rc == 0
        assert pgrep_called == [], "dry-run must not call _pgrep_alive"
        # Original dir must still exist.
        assert adapter_dir.exists()

    def test_alive_pid_without_force_returns_exit_1(self, tmp_path):
        """An alive training PID without --force must return exit code 1."""
        _make_flat_adapter(tmp_path)

        with patch(
            "scripts.migrate.outputs_to_slot_dirs._pgrep_alive",
            return_value=[("12345", "test13_journal_scaffold")],
        ):
            rc = migrate(tmp_path, force=False)

        assert rc == 1

    def test_force_bypasses_alive_pid(self, tmp_path):
        """--force must proceed even when training processes are alive."""
        _make_flat_adapter(tmp_path)

        with patch(
            "scripts.migrate.outputs_to_slot_dirs._pgrep_alive",
            return_value=[("12345", "test13_journal_scaffold")],
        ):
            rc = migrate(tmp_path, force=True)

        assert rc == 0

    def test_no_old_dirs_exits_cleanly(self, tmp_path):
        """When no old-layout dirs exist, exit code is 0."""
        rc = migrate(tmp_path, dry_run=True)
        assert rc == 0
