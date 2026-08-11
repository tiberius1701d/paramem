"""Unit tests for scripts/migrate/outputs_to_slot_dirs.py.

Tests cover:
- Flat-layout adapter → slot layout (weights relocated, meta.json written).
- meta.json has synthesized=True.
- Idempotent rerun is a no-op.
- dry-run writes nothing and skips liveness check.
- Sibling registry SHA-256 is embedded (plaintext hash — see
  TestEncryptedRegistrySha256 for the encrypted-at-rest case).
- Missing sibling registry → UNKNOWN.
- A simhash_registry_*.json sibling is no longer a fallback candidate — it
  is ignored, not hashed.
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

import pytest

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

    def test_simhash_registry_sibling_is_ignored(self, tmp_path):
        """A simhash_registry_*.json beside the adapter is no longer a
        fallback candidate.  It is a different file (a pre-unification
        fingerprint file) whose bytes cannot hash equal to
        indexed_key_registry.json under any hash function, so keeping it as
        a candidate could only ever synthesize a stamp that never matches
        at mount.  With no true registry file present, the result must
        still be UNKNOWN, not the simhash file's hash."""
        from paramem.adapters.manifest import UNKNOWN, read_manifest

        adapter_dir = _make_flat_adapter(tmp_path)
        (tmp_path / "simhash_registry_20260101-000000.json").write_text('{"graph1": 12345}')

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

    def test_grandparent_registry_is_not_a_candidate(self, tmp_path):
        """A registry present ONLY at ``adapter_dir.parent.parent`` (the
        deleted candidate 3) must not be found — it is the adapter ROOT, not
        the tier root a live mount hashes, and its bytes would never match
        ``tier_registry_sha256`` there.  Only ``adapter_dir.parent`` (see
        ``_find_sibling_registry``'s docstring) is checked."""
        from scripts.migrate.outputs_to_slot_dirs import _find_sibling_registry

        run_dir = tmp_path / "run1"
        adapter_dir = _make_flat_adapter(run_dir / "nested")
        (run_dir / "indexed_key_registry.json").write_text('{"active_keys": ["graph1"]}')

        # Sanity: absent at adapter_dir.parent (the only checked location).
        assert not (adapter_dir.parent / "indexed_key_registry.json").exists()
        # Sanity: present at adapter_dir.parent.parent (the deleted candidate).
        assert (adapter_dir.parent.parent / "indexed_key_registry.json").exists()

        result = _find_sibling_registry(adapter_dir, registry_path_override=None)

        from paramem.adapters.manifest import UNKNOWN

        assert result == UNKNOWN

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


class TestEncryptedRegistrySha256:
    """A synthesized manifest's registry_sha256 must hash the sibling
    registry's PLAINTEXT content, not its on-disk (possibly ciphertext)
    bytes — the mount-time comparison (``tier_registry_sha256``,
    paramem/adapters/manifest.py) always hashes plaintext via
    ``plaintext_sha256``, so a ciphertext-based synthesized stamp could
    never match at mount.
    """

    @pytest.fixture(autouse=True)
    def _isolate_daily_identity_cache(self):
        """Isolate the module-level daily-identity cache so this class's
        minted identity never leaks into other tests.

        The load-bearing half is ``_clear_daily_identity_cache()``:
        ``load_daily_identity_cached`` (``paramem/backup/key_store.py``)
        caches the unwrapped identity in a plain module global, keyed by
        nothing, so once loaded it satisfies every later call in the same
        process regardless of env var or ``DAILY_KEY_PATH_DEFAULT`` changes
        — a leaked cache from an earlier test would let a later, unrelated
        test transparently decrypt as this class's identity. The env var
        itself needs no local handling: ``tests/conftest.py``'s autouse
        ``_isolate_paramem_security_env`` already pops
        ``PARAMEM_DAILY_PASSPHRASE`` via ``monkeypatch`` (auto-rolled-back)
        for every test.
        """
        from paramem.backup.key_store import _clear_daily_identity_cache

        _clear_daily_identity_cache()
        yield
        _clear_daily_identity_cache()

    def _mint_daily_identity(self, tmp_path: Path, monkeypatch, passphrase: str = "pw"):
        """Mint a daily identity, point module defaults at it, and return it."""
        from pyrage import x25519

        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )

        daily = mint_daily_identity()
        recovery = x25519.Identity.generate()
        daily_path = tmp_path / "daily_key.age"
        recovery_path = tmp_path / "recovery.pub"
        write_daily_key_file(wrap_daily_identity(daily, passphrase), daily_path)
        write_recovery_pub_file(recovery.to_public(), recovery_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)
        return daily

    def test_synthesized_stamp_over_encrypted_registry_matches_tier_registry_sha256(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Round-trips through the real age-encryption helpers (not a raw
        sha256 of on-disk bytes) so a regression back to ciphertext-hashing
        would fail this test.  The synthesized slot lands directly under
        ``adapter_dir.parent`` (see ``_reshape_dir``) — exactly the tier
        root a live mount hashes via ``tier_registry_sha256`` — so the two
        must agree."""
        from paramem.adapters.manifest import read_manifest, tier_registry_sha256
        from paramem.backup.encryption import AGE_MAGIC, write_infra_bytes

        self._mint_daily_identity(tmp_path, monkeypatch)

        run_dir = tmp_path / "run1"
        adapter_dir = _make_flat_adapter(run_dir)
        registry_path = run_dir / "indexed_key_registry.json"
        plaintext = b'{"active_keys": ["graph1"], "stale": {}, "simhash": {}}'
        write_infra_bytes(registry_path, plaintext)

        # Sanity: on-disk bytes really are age-encrypted.
        on_disk = registry_path.read_bytes()
        assert on_disk.startswith(AGE_MAGIC), "expected age envelope, got plaintext"
        assert on_disk != plaintext

        _reshape_dir(
            adapter_dir,
            registry_path_override=None,
            name_from_config=False,
            dry_run=False,
            verbose=False,
        )
        slots = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) == 1
        manifest = read_manifest(slots[0])

        assert manifest.registry_sha256 == tier_registry_sha256(run_dir)


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

    def test_encrypted_registry_no_identity_available_returns_exit_1(self, tmp_path, caplog):
        """A sibling registry that is an age envelope with no daily identity
        loaded must not propagate a RuntimeError traceback out of migrate()
        -- every other failure path in this script logs an actionable error
        and returns 1; the migrate() boundary now catches this one the same
        way.  The directory itself is left in old layout: its own
        _reshape_dir call raises before the rename runs, so a rerun after
        the identity is loaded picks it back up (rerun-safe)."""
        import logging

        from paramem.backup.encryption import age_encrypt_bytes
        from paramem.backup.key_store import _clear_daily_identity_cache, mint_daily_identity

        _clear_daily_identity_cache()

        adapter_dir = _make_flat_adapter(tmp_path)
        encrypter = mint_daily_identity()
        ciphertext = age_encrypt_bytes(b'{"active_keys":["graph1"]}', [encrypter.to_public()])
        (tmp_path / "indexed_key_registry.json").write_bytes(ciphertext)

        caplog.set_level(logging.ERROR)
        with patch("scripts.migrate.outputs_to_slot_dirs._pgrep_alive", return_value=[]):
            rc = migrate(tmp_path)

        assert rc == 1
        assert any(
            "daily identity" in r.message or "age envelope" in r.message for r in caplog.records
        ), f"expected an actionable error log, got: {[r.message for r in caplog.records]}"
        # The rename never ran -- old-layout dir is untouched.
        assert adapter_dir.exists()

    def test_no_old_dirs_exits_cleanly(self, tmp_path):
        """When no old-layout dirs exist, exit code is 0."""
        rc = migrate(tmp_path, dry_run=True)
        assert rc == 0
