"""Tests for recursive slot sizing in paramem.backup.retention.

Verifies that ``_slot_size_bytes`` (and therefore ``compute_disk_usage``)
correctly sizes slots whose payload files live in subdirectories, as is the
case for ``snapshot_bundle`` slots that store adapter weights under
``adapters/<tier>/adapter_model.safetensors``.

``_slot_size_bytes`` used to iterate only top-level files (``iterdir()``)
and returned ~0 for bundle slots, silently bypassing the disk cap and
retention rules.
"""

from __future__ import annotations

import json
from pathlib import Path

from paramem.backup.retention import _slot_size_bytes, compute_disk_usage
from paramem.server.config import RetentionConfig, RetentionTierConfig, ServerBackupsConfig


def _make_config(max_total_disk_gb: float = 20.0) -> ServerBackupsConfig:
    return ServerBackupsConfig(
        max_total_disk_gb=max_total_disk_gb,
        schedule="daily 04:00",
        artifacts=["config", "graph", "registry"],
        retention=RetentionConfig(
            daily=RetentionTierConfig(keep=7),
            manual=RetentionTierConfig(keep="unlimited", max_disk_gb=5.0),
        ),
    )


def _make_bundle_slot(slot_dir: Path, *, adapter_size_bytes: int = 100_000) -> Path:
    """Create a minimal bundle slot directory with files in subdirectories.

    Simulates the layout produced by ``write_bundle``:
      <slot_dir>/
        bundle.meta.json              (top-level manifest, small)
        config/server.yaml            (small)
        registry/key_metadata.json    (small)
        adapters/episodic/
          adapter_model.safetensors   (large — the bulk of the bundle)
          adapter_config.json         (small)
          meta.json                   (small)

    Returns
    -------
    Path
        The slot directory.
    """
    slot_dir.mkdir(parents=True, exist_ok=True)

    # Top-level bundle manifest
    manifest = {
        "bundle_schema_version": 1,
        "created_at": "2026-05-20T20:55:00Z",
        "tier": "manual",
        "label": None,
        "live_registry_sha256": "a" * 64,
        "base_model": {},
        "files": [],
        "adapters": {},
        "excluded": [],
    }
    (slot_dir / "bundle.meta.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Config subdir
    config_dir = slot_dir / "config"
    config_dir.mkdir()
    (config_dir / "server.yaml").write_bytes(b"model: mistral\n")

    # Registry subdir
    registry_dir = slot_dir / "registry"
    registry_dir.mkdir()
    (registry_dir / "key_metadata.json").write_bytes(b"{}")

    # Adapter subdir with large weight file
    adapter_dir = slot_dir / "adapters" / "episodic"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * adapter_size_bytes)
    (adapter_dir / "adapter_config.json").write_bytes(b'{"lora_type": "lora"}')
    (adapter_dir / "meta.json").write_bytes(b'{"schema_version": 4}')

    return slot_dir


# ---------------------------------------------------------------------------
# _slot_size_bytes: recursive counting
# ---------------------------------------------------------------------------


class TestSlotSizeBytesRecursive:
    def test_top_level_only_slot_sized_correctly(self, tmp_path) -> None:
        """A flat (top-level-only) slot is sized correctly (baseline check)."""
        slot = tmp_path / "flat_slot"
        slot.mkdir()
        (slot / "artifact.bin").write_bytes(b"x" * 1000)
        (slot / "artifact.meta.json").write_bytes(b"y" * 200)
        total = _slot_size_bytes(slot)
        assert total == 1200

    def test_bundle_slot_with_subdir_files_sized_correctly(self, tmp_path) -> None:
        """A bundle slot with large file in subdir is sized correctly.

        Before the fix: _slot_size_bytes returned only the size of
        bundle.meta.json (the only top-level file), ignoring the subdirs.
        After the fix: it recursively counts all files.
        """
        adapter_size = 80_000  # 80 KB simulating weight bytes
        slot = _make_bundle_slot(tmp_path / "bundle_slot", adapter_size_bytes=adapter_size)

        total = _slot_size_bytes(slot)

        # Must be significantly larger than bundle.meta.json alone.
        # bundle.meta.json is small (<1 KB); adapter file is 80 KB.
        assert total > adapter_size, (
            f"Expected total > {adapter_size} (adapter file size), "
            f"got {total}. _slot_size_bytes is likely not recursive."
        )

    def test_nested_files_all_counted(self, tmp_path) -> None:
        """Every file at every nesting level is included in the total."""
        slot = tmp_path / "slot"
        slot.mkdir()
        # Create a deeply nested file
        deep = slot / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "data.bin").write_bytes(b"z" * 5000)
        (slot / "top.bin").write_bytes(b"t" * 1000)

        total = _slot_size_bytes(slot)
        assert total == 6000

    def test_nonexistent_slot_returns_zero(self, tmp_path) -> None:
        """A slot directory that does not exist returns 0 without raising."""
        total = _slot_size_bytes(tmp_path / "nonexistent")
        assert total == 0

    def test_empty_slot_returns_zero(self, tmp_path) -> None:
        """An empty slot directory returns 0."""
        slot = tmp_path / "empty"
        slot.mkdir()
        assert _slot_size_bytes(slot) == 0


# ---------------------------------------------------------------------------
# compute_disk_usage: correctly accounts for bundle slots
# ---------------------------------------------------------------------------


class TestComputeDiskUsageBundleSlots:
    def test_bundle_slot_included_in_total(self, tmp_path) -> None:
        """compute_disk_usage counts bundle slot sizes (recursive)."""
        config = _make_config()
        backups_root = tmp_path / "backups"

        adapter_size = 50_000  # 50 KB
        _make_bundle_slot(
            backups_root / "snapshot" / "20260520-205500",
            adapter_size_bytes=adapter_size,
        )

        # The bundle slot's tier is extracted from bundle.meta.json.
        # compute_disk_usage reads the first *.meta.json in the slot.
        # For bundle slots it should find bundle.meta.json and read "tier".
        usage = compute_disk_usage(backups_root, config, bypass_cache=True)

        # Total must be at least the adapter file size.
        assert usage.total_bytes > adapter_size, (
            f"compute_disk_usage should recursively count bundle slot files; "
            f"total_bytes={usage.total_bytes}, adapter_size={adapter_size}"
        )

    def test_bundle_slot_tier_bucketed(self, tmp_path) -> None:
        """Bundle slot's tier from bundle.meta.json is used for by_tier bucketing."""
        config = _make_config()
        backups_root = tmp_path / "backups"

        _make_bundle_slot(backups_root / "snapshot" / "20260520-205500")

        usage = compute_disk_usage(backups_root, config, bypass_cache=True)
        # The bundle manifest has tier="manual".
        assert "manual" in usage.by_tier, f"Expected 'manual' tier in by_tier; got {usage.by_tier}"

    def test_mixed_slot_types_total(self, tmp_path) -> None:
        """compute_disk_usage totals flat + bundle slots correctly."""
        config = _make_config()
        backups_root = tmp_path / "backups"

        # A regular per-artifact slot (flat).
        flat_slot = backups_root / "config" / "20260520-100000"
        flat_slot.mkdir(parents=True)
        meta = {
            "schema_version": 1,
            "kind": "config",
            "timestamp": "20260520-100000",
            "content_sha256": "abc",
            "size_bytes": 2000,
            "encrypted": False,
            "tier": "daily",
            "label": None,
        }
        (flat_slot / "config-20260520-100000.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        (flat_slot / "config-20260520-100000.bin").write_bytes(b"x" * 2000)

        # A bundle slot with files in subdirs.
        bundle_adapter_size = 30_000
        _make_bundle_slot(
            backups_root / "snapshot" / "20260520-200000",
            adapter_size_bytes=bundle_adapter_size,
        )

        usage = compute_disk_usage(backups_root, config, bypass_cache=True)

        # Total must exceed both flat slot + bundle adapter file.
        assert usage.total_bytes > 2000 + bundle_adapter_size
