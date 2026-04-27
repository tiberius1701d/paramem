"""Unit tests for paramem.adapters.manifest.

Covers:
- Roundtrip write→read, idempotency, UNKNOWN roundtrip, synthesized field.
- Read errors: missing file, malformed JSON, missing required field.
- build_manifest_for: happy path, missing _commit_hash → UNKNOWN,
  registry_path=None/keyed_pairs_path=None → empty hash,
  cache avoids recompute, synthesized always False,
  registry_sha256_override bypasses registry_path.
- atomic_save_adapter: pending→slot promotion, manifest=None omits meta.json,
  PEFT-nested subdir flatten inside pending slot, slot collision bump.
- find_live_slot: empty/missing dir, match, empty-hash matches empty slots,
  skips .pending, unreadable meta skipped, multiple → newest, hash mismatch.
- resolve_adapter_slot: both branches (flat and legacy nested).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    ManifestNotFoundError,
    ManifestSchemaError,
    TokenizerFingerprint,
    build_manifest_for,
    find_live_slot,
    read_manifest,
    resolve_adapter_slot,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_manifest(name: str = "episodic", synthesized: bool = False) -> AdapterManifest:
    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=name,
        trained_at="2026-04-21T00:00:00Z",
        base_model=BaseModelFingerprint(repo="hf/model", sha="abc123", hash="sha256:deadbeef"),
        tokenizer=TokenizerFingerprint(
            name_or_path="hf/model", vocab_size=32000, merges_hash="cafebabe"
        ),
        lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
        registry_sha256="reg123",
        keyed_pairs_sha256="kp456",
        key_count=10,
        synthesized=synthesized,
    )


def _write_slot(base: Path, manifest: AdapterManifest) -> Path:
    """Write a manifest to a new slot subdir and return the slot path."""
    slot = base / manifest.trained_at.replace(":", "").replace("-", "").replace("T", "-")[:15]
    slot.mkdir(parents=True, exist_ok=True)
    write_manifest(slot, manifest)
    return slot


# ---------------------------------------------------------------------------
# 1. Roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_write_read_identical(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()
        write_manifest(slot, m)
        m2 = read_manifest(slot)
        assert m == m2

    def test_idempotent_overwrite(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()
        write_manifest(slot, m)
        write_manifest(slot, m)  # second write
        assert read_manifest(slot) == m

    def test_non_manifest_type_rejected(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        (slot / "meta.json").write_text("[1, 2, 3]")  # array, not object
        with pytest.raises(ManifestSchemaError, match="root must be a JSON object"):
            read_manifest(slot)

    def test_unknown_roundtrips(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo=UNKNOWN, sha=UNKNOWN, hash=UNKNOWN),
            tokenizer=TokenizerFingerprint(
                name_or_path=UNKNOWN, vocab_size=UNKNOWN, merges_hash=UNKNOWN
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=()),
            registry_sha256=UNKNOWN,
            keyed_pairs_sha256=UNKNOWN,
            key_count=UNKNOWN,
            synthesized=True,
        )
        write_manifest(slot, m)
        m2 = read_manifest(slot)
        assert m2.base_model.repo == UNKNOWN
        assert m2.key_count == UNKNOWN
        assert m2.synthesized is True

    def test_synthesized_true_roundtrips(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest(synthesized=True)
        write_manifest(slot, m)
        assert read_manifest(slot).synthesized is True

    def test_synthesized_false_roundtrips(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest(synthesized=False)
        write_manifest(slot, m)
        assert read_manifest(slot).synthesized is False

    def test_absent_synthesized_defaults_to_false(self, tmp_path: Path) -> None:
        """Legacy meta.json without synthesized field must parse with synthesized=False."""
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()
        write_manifest(slot, m)
        # Remove synthesized from on-disk JSON
        raw = json.loads((slot / "meta.json").read_text())
        raw.pop("synthesized", None)
        (slot / "meta.json").write_text(json.dumps(raw))
        assert read_manifest(slot).synthesized is False

    def test_write_raises_if_slot_missing(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            write_manifest(tmp_path / "nonexistent", _minimal_manifest())


# ---------------------------------------------------------------------------
# 2. Read errors
# ---------------------------------------------------------------------------


class TestReadErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        with pytest.raises(ManifestNotFoundError):
            read_manifest(slot)

    def test_malformed_json(self, tmp_path: Path) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        (slot / "meta.json").write_text("{bad json")
        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version",
            "name",
            "trained_at",
            "base_model",
            "tokenizer",
            "lora",
            "registry_sha256",
            "keyed_pairs_sha256",
            "key_count",
        ],
    )
    def test_missing_required_field(self, tmp_path: Path, field: str) -> None:
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()
        write_manifest(slot, m)
        raw = json.loads((slot / "meta.json").read_text())
        raw.pop(field)
        (slot / "meta.json").write_text(json.dumps(raw))
        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

    def test_forward_compat_extra_field(self, tmp_path: Path) -> None:
        """Extra fields in a newer schema version are ignored (forward-compat)."""
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()
        write_manifest(slot, m)
        raw = json.loads((slot / "meta.json").read_text())
        raw["future_field"] = "ignored"
        (slot / "meta.json").write_text(json.dumps(raw))
        read_manifest(slot)  # must not raise


# ---------------------------------------------------------------------------
# 3. build_manifest_for
# ---------------------------------------------------------------------------


class TestBuildManifestFor:
    def _make_model(self, name: str = "hf/base", commit: str | None = "abc") -> MagicMock:
        model = MagicMock()
        model.config._name_or_path = name
        model.config._commit_hash = commit
        model.base_model.model.state_dict.return_value = {}
        peft_cfg = MagicMock()
        peft_cfg.r = 8
        peft_cfg.lora_alpha = 16
        peft_cfg.lora_dropout = 0.0
        peft_cfg.target_modules = ["q_proj", "v_proj"]
        model.peft_config = {"episodic": peft_cfg}
        return model

    def _make_tokenizer(self, name: str = "hf/base") -> MagicMock:
        tok = MagicMock()
        tok.name_or_path = name
        tok.__len__ = lambda self: 32000
        tok.backend_tokenizer.to_str.return_value = '{"model": "bpe"}'
        tok.vocab_file = None
        return tok

    def test_happy_path(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        registry.write_bytes(b'{"active_keys":[]}')
        kp = tmp_path / "keyed_pairs.json"
        kp.write_text('[{"key":"k1"}]')

        m = build_manifest_for(
            self._make_model(),
            self._make_tokenizer(),
            "episodic",
            registry_path=registry,
            keyed_pairs_path=kp,
        )
        assert m.schema_version == MANIFEST_SCHEMA_VERSION
        assert m.name == "episodic"
        assert m.synthesized is False
        assert m.key_count == 1
        assert m.registry_sha256 == hashlib.sha256(b'{"active_keys":[]}').hexdigest()
        assert m.lora.rank == 8
        assert m.lora.alpha == 16

    def test_missing_commit_hash_is_unknown(self, tmp_path: Path) -> None:
        m = build_manifest_for(
            self._make_model(commit=None),
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
        )
        assert m.base_model.sha == UNKNOWN

    def test_registry_path_none_gives_empty_hash(self, tmp_path: Path) -> None:
        m = build_manifest_for(
            self._make_model(),
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
        )
        assert m.registry_sha256 == ""

    def test_keyed_pairs_path_none_gives_empty_hash(self, tmp_path: Path) -> None:
        m = build_manifest_for(
            self._make_model(),
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
        )
        assert m.keyed_pairs_sha256 == ""

    def test_cache_avoids_recompute(self, tmp_path: Path) -> None:
        """Second call with same model + cache must not re-call state_dict."""
        model = self._make_model()
        cache: dict = {}

        build_manifest_for(
            model,
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
            base_model_hash_cache=cache,
        )
        call_count_after_first = model.base_model.model.state_dict.call_count

        build_manifest_for(
            model,
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
            base_model_hash_cache=cache,
        )
        call_count_after_second = model.base_model.model.state_dict.call_count

        assert call_count_after_second == call_count_after_first, (
            "state_dict called again on second build — cache not used"
        )

    def test_synthesized_always_false(self, tmp_path: Path) -> None:
        m = build_manifest_for(
            self._make_model(),
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
        )
        assert m.synthesized is False

    def test_registry_sha256_override_bypasses_path(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        registry.write_bytes(b"irrelevant_content")
        override = "aabbccddeeff0011"

        m = build_manifest_for(
            self._make_model(),
            self._make_tokenizer(),
            "episodic",
            registry_path=registry,  # should be ignored when override present
            keyed_pairs_path=None,
            registry_sha256_override=override,
        )
        assert m.registry_sha256 == override

    def test_key_count_explicit(self, tmp_path: Path) -> None:
        m = build_manifest_for(
            self._make_model(),
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
            key_count=42,
        )
        assert m.key_count == 42

    def test_bfloat16_state_dict_hashes_without_fallback(self, tmp_path: Path) -> None:
        """Regression: numpy lacks bfloat16; build_manifest_for must still hash the weights.

        Pre-fix path returned UNKNOWN with a warning, leaving manifests that the
        startup validator flagged as unknown_fields_in_manifest → PA routing disabled.
        """
        import torch

        model = self._make_model()
        state = {
            "layer.0.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "layer.0.bias": torch.zeros(4, dtype=torch.bfloat16),
        }
        model.base_model.model.state_dict.return_value = state

        m = build_manifest_for(
            model,
            self._make_tokenizer(),
            "episodic",
            registry_path=None,
            keyed_pairs_path=None,
        )
        assert m.base_model.hash != UNKNOWN
        assert m.base_model.hash.startswith("sha256:")
        assert len(m.base_model.hash) == len("sha256:") + 64

    def test_bfloat16_hash_is_deterministic(self, tmp_path: Path) -> None:
        """Identical bfloat16 state dicts must produce identical hashes across calls."""
        import torch

        torch.manual_seed(0)
        weight = torch.randn(8, 8, dtype=torch.bfloat16)
        state = {"layer.0.weight": weight}

        model_a = self._make_model()
        model_a.base_model.model.state_dict.return_value = state
        model_b = self._make_model()
        model_b.base_model.model.state_dict.return_value = state

        m_a = build_manifest_for(
            model_a, self._make_tokenizer(), "episodic", registry_path=None, keyed_pairs_path=None
        )
        m_b = build_manifest_for(
            model_b, self._make_tokenizer(), "episodic", registry_path=None, keyed_pairs_path=None
        )
        assert m_a.base_model.hash == m_b.base_model.hash
        assert m_a.base_model.hash != UNKNOWN


# ---------------------------------------------------------------------------
# 4. atomic_save_adapter
# ---------------------------------------------------------------------------


class TestAtomicSaveAdapter:
    """Tests for the refactored atomic_save_adapter (slot-dir layout)."""

    def _fake_save_nested(self, adapter_name: str):
        """Return a side_effect that writes PEFT-nested layout."""

        def _save(path, selected_adapters):
            nested = Path(path) / adapter_name
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "adapter_model.safetensors").write_bytes(b"weights")
            (nested / "adapter_config.json").write_text("{}")

        return _save

    def _fake_save_flat(self):
        """Return a side_effect that writes flat layout (PEFT >= 0.18 behaviour)."""

        def _save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"weights")
            (Path(path) / "adapter_config.json").write_text("{}")

        return _save

    def _find_slot_dir(self, target: Path) -> Path:
        """Return the first non-hidden subdirectory under *target*."""
        for entry in sorted(target.iterdir()):
            if not entry.name.startswith(".") and entry.is_dir():
                return entry
        raise AssertionError(f"No slot directory found under {target}")

    def test_pending_to_slot_promotion(self, tmp_path: Path) -> None:
        from paramem.models.loader import atomic_save_adapter

        model = MagicMock()
        model.save_pretrained.side_effect = self._fake_save_nested("episodic")
        target = tmp_path / "episodic"
        slot = atomic_save_adapter(model, target, "episodic")

        assert slot is not None
        assert slot.exists()
        assert (slot / "adapter_model.safetensors").exists()
        # Pending dir must be cleaned up after promotion
        assert not (target / ".pending").exists() or not list((target / ".pending").iterdir())

    def test_manifest_none_omits_meta_json(self, tmp_path: Path) -> None:
        from paramem.models.loader import atomic_save_adapter

        model = MagicMock()
        model.save_pretrained.side_effect = self._fake_save_nested("episodic")
        target = tmp_path / "episodic"
        slot = atomic_save_adapter(model, target, "episodic", manifest=None)

        assert slot is not None
        assert slot.exists()
        assert not (slot / "meta.json").exists()

    def test_manifest_written_when_provided(self, tmp_path: Path) -> None:
        from paramem.models.loader import atomic_save_adapter

        model = MagicMock()
        model.save_pretrained.side_effect = self._fake_save_nested("episodic")
        target = tmp_path / "episodic"
        manifest = _minimal_manifest()
        atomic_save_adapter(model, target, "episodic", manifest=manifest)

        from paramem.adapters.manifest import find_live_slot

        slot = find_live_slot(target, manifest.registry_sha256)
        assert slot is not None
        assert (slot / "meta.json").exists()
        assert read_manifest(slot) == manifest

    def test_save_adapter_forwards_to_slot_layout(self, tmp_path: Path) -> None:
        """save_adapter (thin forwarder) must produce slot-dir layout."""
        from paramem.models.loader import save_adapter

        model = MagicMock()
        model.save_pretrained.side_effect = self._fake_save_nested("episodic")
        target = tmp_path / "episodic"
        save_adapter(model, target, "episodic")

        # At least one non-hidden subdirectory must exist under target
        slot_dirs = [e for e in target.iterdir() if not e.name.startswith(".") and e.is_dir()]
        assert len(slot_dirs) == 1, f"Expected one slot dir, got: {slot_dirs}"
        assert (slot_dirs[0] / "adapter_model.safetensors").exists()


class TestPeftNestedSubdirFlattenInsidePendingSlot:
    """The PEFT-nested subdir flatten must happen INSIDE .pending/<ts>/ before rename."""

    def test_flat_layout_before_outer_rename(self, tmp_path: Path) -> None:
        """Verify: inside pending slot, files are flat; outer rename produces flat slot."""
        from paramem.models.loader import atomic_save_adapter

        pending_flat_state: dict = {}

        def _fake_save_nested(path, selected_adapters):
            # PEFT writes nested: <path>/episodic/adapter_*.*
            nested = Path(path) / "episodic"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "adapter_model.safetensors").write_bytes(b"w")
            (nested / "adapter_config.json").write_text("{}")

        original_rename = Path.rename

        def _intercept_rename(self_path, target):
            # Only intercept the outer rename from .pending/<ts>/ to target/<ts>/
            # The outer rename is a directory (not a file) and its parent is .pending
            if (
                ".pending" in str(self_path)
                and self_path.is_dir()
                and self_path.parent.name == ".pending"
            ):
                # Record what files are in the pending slot at rename time
                pending_flat_state["files"] = [f.name for f in self_path.iterdir()]
                pending_flat_state["has_nested"] = (self_path / "episodic").exists()
            return original_rename(self_path, target)

        model = MagicMock()
        model.save_pretrained.side_effect = _fake_save_nested

        with patch.object(Path, "rename", _intercept_rename):
            atomic_save_adapter(model, tmp_path / "episodic", "episodic")

        # The nested dir must be absent when the outer rename fires
        assert not pending_flat_state.get("has_nested", True), (
            "Flatten did not run inside pending slot before outer rename"
        )
        # Adapter files must be present at the flat level
        assert "adapter_model.safetensors" in pending_flat_state.get("files", [])

    def test_fallback_when_peft_already_flat(self, tmp_path: Path) -> None:
        """When PEFT writes flat (no nested subdir), flatten is a no-op."""
        from paramem.models.loader import atomic_save_adapter

        def _fake_save_flat(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"w")
            (Path(path) / "adapter_config.json").write_text("{}")

        model = MagicMock()
        model.save_pretrained.side_effect = _fake_save_flat

        slot = atomic_save_adapter(model, tmp_path / "episodic", "episodic")

        assert slot is not None
        assert slot.exists()
        assert (slot / "adapter_model.safetensors").exists()
        # No nested subdir in final slot
        assert not (slot / "episodic").exists()


# ---------------------------------------------------------------------------
# 5. find_live_slot
# ---------------------------------------------------------------------------


class TestFindLiveSlot:
    def _write_slot_with_hash(self, base: Path, ts: str, reg_hash: str) -> Path:
        slot = base / ts
        slot.mkdir(parents=True)
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo="r", sha="s", hash="h"),
            tokenizer=TokenizerFingerprint(name_or_path="t", vocab_size=1, merges_hash="m"),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=()),
            registry_sha256=reg_hash,
            keyed_pairs_sha256="kp",
            key_count=1,
        )
        write_manifest(slot, m)
        return slot

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        assert find_live_slot(tmp_path, "abc") is None

    def test_missing_dir_returns_none(self, tmp_path: Path) -> None:
        assert find_live_slot(tmp_path / "nonexistent", "abc") is None

    def test_match_returns_slot(self, tmp_path: Path) -> None:
        slot = self._write_slot_with_hash(tmp_path, "20260421-000000", "abc123")
        assert find_live_slot(tmp_path, "abc123") == slot

    def test_empty_hash_matches_empty_registry_slot(self, tmp_path: Path) -> None:
        slot = self._write_slot_with_hash(tmp_path, "20260421-000000", "")
        assert find_live_slot(tmp_path, "") == slot

    def test_skips_pending(self, tmp_path: Path) -> None:
        pending = tmp_path / ".pending"
        pending.mkdir()
        self._write_slot_with_hash(pending, "20260421-000000", "abc")
        assert find_live_slot(tmp_path, "abc") is None

    def test_unreadable_meta_skipped(self, tmp_path: Path) -> None:
        slot = tmp_path / "20260421-000000"
        slot.mkdir()
        (slot / "meta.json").write_text("{bad json")
        assert find_live_slot(tmp_path, "") is None

    def test_multiple_matches_newest_wins(self, tmp_path: Path) -> None:
        import os

        s1 = self._write_slot_with_hash(tmp_path, "20260421-000001", "match")
        s2 = self._write_slot_with_hash(tmp_path, "20260421-000002", "match")
        # Force s2's mtime to be 10 seconds after s1
        t_s1 = s1.stat().st_mtime
        os.utime(s1, (t_s1, t_s1))
        os.utime(s2, (t_s1 + 10, t_s1 + 10))
        result = find_live_slot(tmp_path, "match")
        assert result == s2

    def test_hash_mismatch_returns_none(self, tmp_path: Path) -> None:
        self._write_slot_with_hash(tmp_path, "20260421-000000", "old_hash")
        assert find_live_slot(tmp_path, "new_hash") is None


# ---------------------------------------------------------------------------
# 6. resolve_adapter_slot
# ---------------------------------------------------------------------------


class TestResolveAdapterSlot:
    def _write_slot(self, base: Path, ts: str, reg_hash: str) -> Path:
        slot = base / ts
        slot.mkdir(parents=True)
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo="r", sha="s", hash="h"),
            tokenizer=TokenizerFingerprint(name_or_path="t", vocab_size=1, merges_hash="m"),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=()),
            registry_sha256=reg_hash,
            keyed_pairs_sha256="kp",
            key_count=1,
        )
        write_manifest(slot, m)
        return slot

    def test_flat_layout(self, tmp_path: Path) -> None:
        """New flat layout: <base_dir>/<ts>/ contains adapter files."""
        base = tmp_path / "episodic"
        base.mkdir()
        slot = self._write_slot(base, "20260421-000000", "hashval")
        result = resolve_adapter_slot(base, "episodic", "hashval")
        assert result == slot

    def test_legacy_nested_layout(self, tmp_path: Path) -> None:
        """Legacy pre-migration layout: <base_dir>/episodic/<ts>/."""
        base = tmp_path / "episodic"
        nested = base / "episodic"
        nested.mkdir(parents=True)
        slot = self._write_slot(nested, "20260421-000000", "hashval")
        result = resolve_adapter_slot(base, "episodic", "hashval")
        assert result == slot

    def test_empty_hash_scoped(self, tmp_path: Path) -> None:
        """Empty live_hash matches empty-registry slots; sibling files ignored."""
        base = tmp_path / "episodic"
        base.mkdir()
        # Add a sibling file that is not a slot
        (base / "state.json").write_text("{}")
        slot = self._write_slot(base, "20260421-000001", "")
        result = resolve_adapter_slot(base, "episodic", "")
        assert result == slot
        assert result != base / "state.json"

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        base = tmp_path / "episodic"
        base.mkdir()
        result = resolve_adapter_slot(base, "episodic", "missinghash")
        assert result is None
