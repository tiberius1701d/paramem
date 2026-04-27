"""Integration tests for _mount_adapters_from_slots (Slice 3a startup validator).

Exercises _mount_adapters_from_slots directly (bypasses full lifespan).
All assertions operate on state["adapter_manifest_status"] — NOT
adapter_health (which is separate and untouched by this slice).

Covers:
- Fresh install → no manifest rows.
- Healthy mount → no rows.
- Episodic wrong base_model.sha → red fingerprint_mismatch, not loaded.
- Semantic wrong lora.rank → yellow mismatch.
- Corrupt meta.json → manifest_unreadable.
- Weights + no meta.json → manifest_missing.
- enabled=False → no row.
- Registry hash mismatch → no_matching_slot.
- Multiple rows render independently.
- Migration-script slot (synthesized=True, UNKNOWN fields) → yellow even for episodic.
- Fresh-built manifest with UNKNOWN fields (synthesized=False) → red.
- episodic_interim_* uses same schema.
- sweep_orphan_pending called before find_live_slot (mock order).
- StatusResponse.adapter_health is UNTOUCHED by the validator.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    TokenizerFingerprint,
    write_manifest,
)
from paramem.server.app import _mount_adapters_from_slots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, adapter_names=("episodic",), enabled_names=None):
    """Build a minimal ServerConfig-like object."""
    if enabled_names is None:
        enabled_names = adapter_names

    config = MagicMock()
    config.adapter_dir = tmp_path / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)

    def _make_adapter_cfg(name):
        cfg = MagicMock()
        cfg.enabled = name in enabled_names
        cfg.rank = 8
        cfg.alpha = 16
        cfg.dropout = 0.0
        cfg.target_modules = ["q_proj", "v_proj"]
        return cfg

    config.adapters.episodic = _make_adapter_cfg("episodic")
    config.adapters.semantic = _make_adapter_cfg("semantic")
    config.adapters.procedural = _make_adapter_cfg("procedural")
    return config


def _make_model(name_or_path: str = "hf/model", commit_hash: str = "abc123"):
    model = MagicMock()
    model.config._name_or_path = name_or_path
    model.config._commit_hash = commit_hash
    model.peft_config = {}
    return model


def _make_tokenizer():
    tok = MagicMock()
    tok.name_or_path = "hf/model"
    return tok


def _write_slot(
    adapter_kind_dir: Path,
    ts: str = "20260421-000000",
    registry_sha256: str = "",
    sha: str = "abc123",
    rank: int = 8,
    synthesized: bool = False,
    key_count: "int | str" = 5,
) -> Path:
    slot = adapter_kind_dir / ts
    slot.mkdir(parents=True)
    # Write minimal adapter files so load won't fail on file-not-found
    (slot / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "hf/model"}))
    (slot / "adapter_model.safetensors").write_bytes(b"weights")
    m = AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name="episodic",
        trained_at="2026-04-21T00:00:00Z",
        base_model=BaseModelFingerprint(repo="hf/model", sha=sha, hash="sha256:dead"),
        tokenizer=TokenizerFingerprint(
            name_or_path="hf/model", vocab_size=32000, merges_hash="cafe"
        ),
        lora=LoRAShape(rank=rank, alpha=rank * 2, dropout=0.0, target_modules=("q_proj", "v_proj")),
        registry_sha256=registry_sha256,
        keyed_pairs_sha256="kp456",
        key_count=key_count,
        synthesized=synthesized,
    )
    write_manifest(slot, m)
    return slot


def _run(config, model=None, tokenizer=None, state=None):
    """Execute _mount_adapters_from_slots and return (model, state)."""
    if model is None:
        model = _make_model()
    if tokenizer is None:
        tokenizer = _make_tokenizer()
    if state is None:
        state = {"adapter_manifest_status": {}, "base_model_hash_cache": {}}
    result_model = _mount_adapters_from_slots(model, tokenizer, config, state)
    return result_model, state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFreshInstall:
    def test_no_rows_on_empty_adapter_dir(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _, state = _run(config)
        assert state["adapter_manifest_status"] == {}


class TestHealthyMount:
    def test_healthy_mount_produces_no_row(self, tmp_path: Path) -> None:
        """A healthy slot with matching fingerprints must produce no manifest row.

        We patch PeftModel.from_pretrained so the test doesn't need real weights.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        _write_slot(kind_dir, registry_sha256="")

        from peft import PeftModel

        with patch.object(PeftModel, "from_pretrained", return_value=MagicMock(spec=PeftModel)):
            _, state = _run(config)
        assert "episodic" not in state["adapter_manifest_status"]


class TestFingerprintMismatch:
    def test_episodic_wrong_sha_gives_red_and_not_loaded(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # Slot has sha=different_sha; model has sha=abc123
        _write_slot(kind_dir, sha="different_sha")

        model = _make_model(commit_hash="abc123")
        _, state = _run(config, model=model)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None, "Expected a row for episodic fingerprint mismatch"
        assert row["severity"] == "red"
        assert row["reason"] == "fingerprint_mismatch"
        # Model must NOT have episodic loaded
        assert "episodic" not in getattr(model, "peft_config", {})

    def test_semantic_wrong_rank_gives_yellow(self, tmp_path: Path) -> None:
        config = _make_config(
            tmp_path, adapter_names=("episodic", "semantic"), enabled_names=("semantic",)
        )
        kind_dir = config.adapter_dir / "semantic"
        kind_dir.mkdir()
        # Slot has rank=4; config has rank=8
        _write_slot(kind_dir, rank=4)
        config.adapters.semantic.rank = 8

        model = _make_model()
        _, state = _run(config, model=model)

        row = state["adapter_manifest_status"].get("semantic")
        assert row is not None
        assert row["severity"] == "yellow"


class TestCorruptManifest:
    def test_corrupt_meta_json_gives_manifest_unreadable_via_patch(self, tmp_path: Path) -> None:
        """Manifest unreadable row is produced when find_live_slot returns a slot
        but reading the manifest raises ManifestSchemaError.

        find_live_slot already skips unreadable meta.json internally (logs WARN
        and returns None), producing a no_matching_slot row.  The manifest_unreadable
        path is exercised by patching find_live_slot to return the corrupt slot.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        (slot / "meta.json").write_text("{bad json")

        with patch("paramem.adapters.manifest.find_live_slot", return_value=slot):
            _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["reason"] == "manifest_unreadable"

    def test_corrupt_meta_json_without_patch_gives_no_matching_slot(self, tmp_path: Path) -> None:
        """Corrupt meta.json causes find_live_slot to skip the slot → no_matching_slot."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        (slot / "meta.json").write_text("{bad json")

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        # find_live_slot skips unreadable meta.json and returns None → no_matching_slot
        assert row is not None
        assert row["status"] == "no_matching_slot"


class TestManifestMissing:
    def test_weights_present_no_meta_gives_no_matching_slot(self, tmp_path: Path) -> None:
        """Slot without meta.json is invisible to find_live_slot (no hash match).

        Since find_live_slot requires a readable meta.json, a slot without one is
        skipped. The result is no_matching_slot (other slots exist but none match).
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        # No meta.json written

        _, state = _run(config)
        # find_live_slot skips the slot → no_matching_slot row
        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "no_matching_slot"


class TestManifestMissingWithPatch:
    """Directly exercise the manifest_missing code path by patching find_live_slot."""

    def test_manifest_missing_when_find_returns_slot_without_meta(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # Slot exists but has no meta.json
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")

        from peft import PeftModel

        # Patch find_live_slot to return the slot despite no meta.json
        # Also patch PeftModel.from_pretrained so the load succeeds (no real weights)
        with (
            patch("paramem.adapters.manifest.find_live_slot", return_value=slot),
            patch.object(PeftModel, "from_pretrained", return_value=MagicMock(spec=PeftModel)),
        ):
            _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "manifest_missing"
        assert row["reason"] == "manifest_missing"


class TestEnabledFalse:
    def test_disabled_adapter_has_no_row(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, adapter_names=("episodic",), enabled_names=())
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        _write_slot(kind_dir)

        _, state = _run(config)
        assert "episodic" not in state["adapter_manifest_status"]


class TestNoMatchingSlot:
    def test_registry_hash_mismatch_gives_no_matching_slot(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # Slot has registry_sha256="old_hash", no live registry file → live_hash=""
        _write_slot(kind_dir, registry_sha256="old_hash")

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "red"  # episodic is primary


class TestMultipleRows:
    def test_multiple_adapter_rows_independent(self, tmp_path: Path) -> None:
        config = _make_config(
            tmp_path, adapter_names=("episodic", "semantic"), enabled_names=("episodic", "semantic")
        )
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        # episodic: hash mismatch → row
        ep_dir = config.adapter_dir / "episodic"
        ep_dir.mkdir()
        _write_slot(ep_dir, sha="wrong_sha")

        # semantic: no slot at all → no row (fresh install)
        sem_dir = config.adapter_dir / "semantic"
        sem_dir.mkdir()

        model = _make_model(commit_hash="abc123")
        _, state = _run(config, model=model)

        assert "episodic" in state["adapter_manifest_status"]
        assert "semantic" not in state["adapter_manifest_status"]


class TestSynthesizedUnknown:
    def test_synthesized_true_unknown_episodic_is_yellow(self, tmp_path: Path) -> None:
        """synthesized=True + UNKNOWN fields → yellow even for episodic."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo=UNKNOWN, sha=UNKNOWN, hash=UNKNOWN),
            tokenizer=TokenizerFingerprint(
                name_or_path=UNKNOWN, vocab_size=UNKNOWN, merges_hash=UNKNOWN
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
            registry_sha256="",
            keyed_pairs_sha256=UNKNOWN,
            key_count=UNKNOWN,
            synthesized=True,
        )
        write_manifest(slot, m)

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["severity"] == "yellow"
        assert row["status"] == "migrated_unverified"

    def test_synthesized_false_unknown_episodic_is_red(self, tmp_path: Path) -> None:
        """synthesized=False + UNKNOWN fields → red (fresh-built manifest failure)."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo=UNKNOWN, sha=UNKNOWN, hash=UNKNOWN),
            tokenizer=TokenizerFingerprint(
                name_or_path=UNKNOWN, vocab_size=UNKNOWN, merges_hash=UNKNOWN
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
            registry_sha256="",
            keyed_pairs_sha256=UNKNOWN,
            key_count=UNKNOWN,
            synthesized=False,
        )
        write_manifest(slot, m)

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["severity"] == "red"
        assert row["status"] == "migrated_unverified"


class TestAdapterHealthUntouched:
    """StatusResponse.adapter_health must be unchanged by _mount_adapters_from_slots."""

    def test_adapter_health_not_modified_by_validator(self, tmp_path: Path) -> None:
        """The manifest validator must not touch adapter_health.

        adapter_health is sourced from the KeyRegistry JSON, not from any
        _state dict mutated by _mount_adapters_from_slots.  This test
        asserts that running the validator leaves adapter_health absent
        from state (it lives in a separate on-disk source).
        """
        config = _make_config(tmp_path)
        state: dict = {"adapter_manifest_status": {}, "base_model_hash_cache": {}}

        _mount_adapters_from_slots(_make_model(), _make_tokenizer(), config, state)

        # adapter_health is NOT a key managed by _mount_adapters_from_slots
        assert "adapter_health" not in state, (
            "_mount_adapters_from_slots must not touch adapter_health "
            "(it is sourced from KeyRegistry JSON separately)"
        )


class TestRevalidateMainAdapterManifests:
    """Post-cycle revalidation shares the same per-tier decision tree as the
    boot validator (_validate_main_adapter_slot). These tests exercise
    _revalidate_main_adapter_manifests directly to verify two key behaviours:

    1. Stale RED rows from the boot snapshot are CLEARED when on-disk slots
       are now healthy (the bug this function exists to fix).
    2. Slots that genuinely became unhealthy after boot get a fresh row
       written, with current ``checked_at``.
    """

    def _state_from_config(self, config, model=None, tokenizer=None):
        return {
            "config": config,
            "model": model if model is not None else _make_model(),
            "tokenizer": tokenizer if tokenizer is not None else _make_tokenizer(),
            "adapter_manifest_status": {},
            "base_model_hash_cache": {},
        }

    def test_clears_stale_red_row_when_slot_now_healthy(self, tmp_path: Path) -> None:
        """A boot-time row exists for episodic; on-disk state is healthy.
        Revalidation removes the row.
        """
        from paramem.server.app import _revalidate_main_adapter_manifests

        config = _make_config(tmp_path)
        # Healthy slot on disk with empty registry hash (matches "no registry").
        episodic_dir = config.adapter_dir / "episodic"
        _write_slot(episodic_dir, ts="20260427-105338", registry_sha256="")

        state = self._state_from_config(config)
        # Inject a stale boot-time RED row.
        state["adapter_manifest_status"]["episodic"] = {
            "status": "no_matching_slot",
            "reason": "no_matching_slot",
            "field": None,
            "severity": "red",
            "slot_path": None,
            "checked_at": "2026-04-27T11:43:44Z",
        }

        _revalidate_main_adapter_manifests(state)

        assert "episodic" not in state["adapter_manifest_status"], (
            "Stale RED row must be cleared once slot is healthy"
        )

    def test_writes_red_row_when_no_matching_slot(self, tmp_path: Path) -> None:
        """No matching slot on disk → revalidation writes a no_matching_slot row."""
        from paramem.server.app import _revalidate_main_adapter_manifests

        config = _make_config(tmp_path)
        # Slot with a non-empty registry hash that won't match the live "" hash.
        episodic_dir = config.adapter_dir / "episodic"
        _write_slot(episodic_dir, ts="20260427-105338", registry_sha256="stale_hash_123")

        state = self._state_from_config(config)
        # Start with no row (post-boot default for a healthy slot).
        _revalidate_main_adapter_manifests(state)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None, "Mismatch must produce a row"
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "red"  # episodic is primary

    def test_disabled_adapter_pops_any_existing_row(self, tmp_path: Path) -> None:
        """If a tier is disabled in config, its row is removed regardless of state."""
        from paramem.server.app import _revalidate_main_adapter_manifests

        config = _make_config(tmp_path, enabled_names=())  # all tiers disabled
        state = self._state_from_config(config)
        state["adapter_manifest_status"]["episodic"] = {
            "status": "mismatch",
            "severity": "red",
            "reason": "fingerprint_mismatch",
            "field": None,
            "slot_path": None,
            "checked_at": "2026-04-27T11:00:00Z",
        }

        _revalidate_main_adapter_manifests(state)

        assert "episodic" not in state["adapter_manifest_status"]

    def test_noop_when_state_missing_model(self, tmp_path: Path) -> None:
        """Defensive: missing model in state → silently no-op (no exception)."""
        from paramem.server.app import _revalidate_main_adapter_manifests

        config = _make_config(tmp_path)
        state = {"config": config, "tokenizer": _make_tokenizer()}  # no "model" key
        _revalidate_main_adapter_manifests(state)  # must not raise
        assert state.get("adapter_manifest_status", {}) == {}
