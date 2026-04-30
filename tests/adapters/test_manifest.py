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
    _hash_safetensors_files,
    _lookup_hash_from_manifests,
    _resolve_base_safetensors,
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
        """Second call with same model + cache must return the cached hash.

        The cache is populated on first call (via read-back or file-hash);
        the second call must hit the cache and NOT invoke _resolve_base_safetensors
        or _lookup_hash_from_manifests again.
        """
        model = self._make_model()
        cache: dict = {}

        with patch(
            "paramem.adapters.manifest._resolve_base_safetensors", return_value=None
        ) as mock_resolve:
            build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                base_model_hash_cache=cache,
            )
            calls_after_first = mock_resolve.call_count

            build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                base_model_hash_cache=cache,
            )
            calls_after_second = mock_resolve.call_count

        assert calls_after_second == calls_after_first, (
            "_resolve_base_safetensors called again on second build — cache not used"
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

    def test_file_hash_returns_sha256_when_safetensors_resolved(self, tmp_path: Path) -> None:
        """build_manifest_for returns a sha256 hash when safetensors files are resolved.

        Replaces the old bfloat16 state_dict regression test: the new cold path
        uses mmap file-hashing rather than state_dict walks, so this test patches
        _resolve_base_safetensors to return a real file and verifies the hash format.
        """
        import safetensors.torch
        import torch

        weight_file = tmp_path / "model.safetensors"
        safetensors.torch.save_file({"weight": torch.zeros(4, 4)}, str(weight_file))

        model = self._make_model()

        with patch(
            "paramem.adapters.manifest._resolve_base_safetensors",
            return_value=[weight_file],
        ):
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

    def test_file_hash_is_deterministic(self, tmp_path: Path) -> None:
        """Identical safetensors files must produce identical hashes across model instances."""
        import safetensors.torch
        import torch

        torch.manual_seed(0)
        weight_file = tmp_path / "model.safetensors"
        safetensors.torch.save_file({"weight": torch.randn(8, 8)}, str(weight_file))

        model_a = self._make_model()
        model_b = self._make_model()

        with patch(
            "paramem.adapters.manifest._resolve_base_safetensors",
            return_value=[weight_file],
        ):
            m_a = build_manifest_for(
                model_a,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
            )
            m_b = build_manifest_for(
                model_b,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
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


# ---------------------------------------------------------------------------
# 7. write_manifest atomic write semantics
# ---------------------------------------------------------------------------


class TestWriteManifestAtomicWrite:
    """write_manifest must route through _atomic_write_bytes for fsync safety."""

    def test_uses_atomic_helper(self, tmp_path: Path) -> None:
        """_atomic_write_bytes is called exactly once with the meta.json path."""
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()

        recorded: list[tuple] = []

        def _fake_atomic(path, body: bytes) -> None:
            recorded.append((path, body))

        with patch(
            "paramem.backup.encryption._atomic_write_bytes",
            side_effect=_fake_atomic,
        ):
            write_manifest(slot, m)

        assert len(recorded) == 1, "_atomic_write_bytes must be called exactly once"
        assert recorded[0][0] == slot / "meta.json", f"Wrong path: {recorded[0][0]!r}"
        # Bytes must be valid UTF-8 JSON
        import json as _json

        parsed = _json.loads(recorded[0][1].decode("utf-8"))
        assert parsed["name"] == m.name

    def test_no_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        """When _atomic_write_bytes raises, OSError propagates and meta.json is absent."""
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()

        with patch(
            "paramem.backup.encryption._atomic_write_bytes",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError, match="disk full"):
                write_manifest(slot, m)

        # The helper guarantees no partial .tmp on OSError, and canonical path absent.
        assert not (slot / "meta.json").exists(), "meta.json must not exist after write failure"

    def test_overwrites_atomically(self, tmp_path: Path) -> None:
        """Writing manifest B after manifest A leaves only manifest B on disk."""
        slot = tmp_path / "slot0"
        slot.mkdir()
        m_a = _minimal_manifest(name="episodic")
        m_b = _minimal_manifest(name="semantic")

        write_manifest(slot, m_a)
        write_manifest(slot, m_b)

        m_disk = read_manifest(slot)
        assert m_disk.name == "semantic", (
            f"Expected 'semantic' after overwrite but got '{m_disk.name}'"
        )

    def test_no_tmp_left_behind(self, tmp_path: Path) -> None:
        """After a successful write_manifest, no .tmp files remain in the slot."""
        slot = tmp_path / "slot0"
        slot.mkdir()
        m = _minimal_manifest()

        write_manifest(slot, m)

        tmp_files = list(slot.glob("*.tmp"))
        assert not tmp_files, (
            f"Leftover .tmp files after write_manifest: {[str(p) for p in tmp_files]}"
        )


# ---------------------------------------------------------------------------
# 8. _hash_safetensors_files
# ---------------------------------------------------------------------------


class TestHashSafetensorsFiles:
    def test_file_hash_pinned(self, tmp_path: Path) -> None:
        """Hash of two safetensors files must equal SHA-256 of their concatenated bytes."""
        import safetensors.torch
        import torch

        f1 = tmp_path / "shard1.safetensors"
        f2 = tmp_path / "shard2.safetensors"
        # Use deterministic tensors so reference hash is reproducible
        torch.manual_seed(42)
        safetensors.torch.save_file({"a": torch.ones(4)}, str(f1))
        safetensors.torch.save_file({"b": torch.zeros(4)}, str(f2))

        result = _hash_safetensors_files([f1, f2])

        # Reference: SHA-256 of the concatenated raw bytes in order
        expected = "sha256:" + hashlib.sha256(f1.read_bytes() + f2.read_bytes()).hexdigest()
        assert result == expected

    def test_single_file(self, tmp_path: Path) -> None:
        """Single-file case matches direct SHA-256 of that file."""
        import safetensors.torch
        import torch

        f = tmp_path / "model.safetensors"
        safetensors.torch.save_file({"w": torch.eye(3)}, str(f))

        result = _hash_safetensors_files([f])
        expected = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
        assert result == expected

    def test_order_matters(self, tmp_path: Path) -> None:
        """Swapping file order must produce a different hash."""
        import safetensors.torch
        import torch

        f1 = tmp_path / "a.safetensors"
        f2 = tmp_path / "b.safetensors"
        safetensors.torch.save_file({"x": torch.ones(4)}, str(f1))
        safetensors.torch.save_file({"y": torch.zeros(4)}, str(f2))

        hash_ab = _hash_safetensors_files([f1, f2])
        hash_ba = _hash_safetensors_files([f2, f1])
        assert hash_ab != hash_ba


# ---------------------------------------------------------------------------
# 9. _lookup_hash_from_manifests
# ---------------------------------------------------------------------------


def _write_meta(base: Path, tier: str, slot_ts: str, manifest: AdapterManifest) -> Path:
    """Write manifest to <base>/<tier>/<slot_ts>/meta.json, creating dirs."""
    slot = base / tier / slot_ts
    slot.mkdir(parents=True, exist_ok=True)
    write_manifest(slot, manifest)
    return slot


def _make_manifest_with_base(
    repo: str,
    sha: str,
    base_hash: str,
    trained_at: str = "2026-04-21T00:00:00Z",
) -> AdapterManifest:
    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name="episodic",
        trained_at=trained_at,
        base_model=BaseModelFingerprint(repo=repo, sha=sha, hash=base_hash),
        tokenizer=TokenizerFingerprint(name_or_path="t", vocab_size=1, merges_hash="m"),
        lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=()),
        registry_sha256="reg",
        keyed_pairs_sha256="kp",
        key_count=1,
    )


class TestLookupHashFromManifests:
    def test_returns_none_when_root_empty(self, tmp_path: Path) -> None:
        """Empty adapter_root returns None without error."""
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result is None

    def test_returns_none_when_root_missing(self, tmp_path: Path) -> None:
        """Non-existent adapter_root returns None without error."""
        result = _lookup_hash_from_manifests(tmp_path / "nope", "hf/model", "abc123")
        assert result is None

    def test_returns_none_for_unknown_commit_sha(self, tmp_path: Path) -> None:
        """UNKNOWN commit_sha always returns None (no reliable match key)."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/model", "abc123", "sha256:dead"),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", UNKNOWN)
        assert result is None

    def test_returns_hash_on_match(self, tmp_path: Path) -> None:
        """Returns base_model.hash from a slot matching (repo, sha)."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/model", "abc123", "sha256:cafecafe"),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result == "sha256:cafecafe"

    def test_skips_unknown_slot_hash(self, tmp_path: Path) -> None:
        """Slots whose base_model.hash == UNKNOWN are excluded from results."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/model", "abc123", UNKNOWN),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result is None

    def test_repo_mismatch_returns_none(self, tmp_path: Path) -> None:
        """A slot with matching sha but different repo does not match."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/OTHER", "abc123", "sha256:dead"),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result is None

    def test_sha_mismatch_returns_none(self, tmp_path: Path) -> None:
        """A slot with matching repo but different sha does not match."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/model", "DIFFERENT", "sha256:dead"),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result is None

    def test_skips_pending_dir(self, tmp_path: Path) -> None:
        """Slots inside .pending are skipped."""
        pending = tmp_path / "episodic" / ".pending"
        pending.mkdir(parents=True, exist_ok=True)
        slot = pending / "20260421-000000"
        slot.mkdir()
        write_manifest(
            slot,
            _make_manifest_with_base("hf/model", "abc123", "sha256:cafecafe"),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result is None

    def test_newest_trained_at_wins(self, tmp_path: Path) -> None:
        """When multiple slots match, the one with the newest trained_at wins."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000001",
            _make_manifest_with_base(
                "hf/model",
                "abc123",
                "sha256:old",
                trained_at="2026-04-20T00:00:00Z",
            ),
        )
        _write_meta(
            tmp_path,
            "semantic",
            "20260421-000002",
            _make_manifest_with_base(
                "hf/model",
                "abc123",
                "sha256:new",
                trained_at="2026-04-21T00:00:00Z",
            ),
        )
        result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        assert result == "sha256:new"

    def test_disagreeing_hashes_warns_and_picks_newest(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two matching slots with different hashes — warns and returns the newer hash."""
        import logging

        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000001",
            _make_manifest_with_base(
                "hf/model",
                "abc123",
                "sha256:older_hash",
                trained_at="2026-04-19T00:00:00Z",
            ),
        )
        _write_meta(
            tmp_path,
            "semantic",
            "20260421-000002",
            _make_manifest_with_base(
                "hf/model",
                "abc123",
                "sha256:newer_hash",
                trained_at="2026-04-21T00:00:00Z",
            ),
        )

        # caplog.at_level alone does not capture in this project (log propagation
        # is intercepted); attach the handler directly to the named logger.
        named_logger = logging.getLogger("paramem.adapters.manifest")
        named_logger.addHandler(caplog.handler)
        try:
            result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        finally:
            named_logger.removeHandler(caplog.handler)

        assert result == "sha256:newer_hash"
        assert any(
            "sha256:older_hash" in r.message and "sha256:newer_hash" in r.message
            for r in caplog.records
        ), f"Expected warn with both hashes; got: {[r.message for r in caplog.records]}"

    def test_same_hash_across_slots_no_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two matching slots with the SAME hash must NOT emit a warning."""
        import logging

        for ts in ("20260421-000001", "20260421-000002"):
            _write_meta(
                tmp_path,
                "episodic",
                ts,
                _make_manifest_with_base("hf/model", "abc123", "sha256:consistent"),
            )

        named_logger = logging.getLogger("paramem.adapters.manifest")
        named_logger.addHandler(caplog.handler)
        try:
            result = _lookup_hash_from_manifests(tmp_path, "hf/model", "abc123")
        finally:
            named_logger.removeHandler(caplog.handler)

        assert result == "sha256:consistent"
        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warn_records, (
            f"Unexpected warnings with matching hashes: {[r.message for r in warn_records]}"
        )


# ---------------------------------------------------------------------------
# 10. build_manifest_for — new warm/cold path integration
# ---------------------------------------------------------------------------


class TestBuildManifestForHashPaths:
    """Tests for the read-back and file-hash paths in build_manifest_for."""

    def _make_model(self, name: str = "hf/base", commit: str = "abc123") -> MagicMock:
        model = MagicMock()
        model.config._name_or_path = name
        model.config._commit_hash = commit
        peft_cfg = MagicMock()
        peft_cfg.r = 8
        peft_cfg.lora_alpha = 16
        peft_cfg.lora_dropout = 0.0
        peft_cfg.target_modules = ["q_proj", "v_proj"]
        model.peft_config = {"episodic": peft_cfg}
        return model

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.name_or_path = "hf/base"
        tok.__len__ = lambda self: 32000
        tok.backend_tokenizer.to_str.return_value = '{"model": "bpe"}'
        tok.vocab_file = None
        return tok

    def test_manifest_readback_skips_file_hash(self, tmp_path: Path) -> None:
        """When adapter_root has a matching slot, _hash_safetensors_files must NOT be called."""
        # Pre-write a slot with known hash
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/base", "abc123", "sha256:from_readback"),
        )

        model = self._make_model(name="hf/base", commit="abc123")

        with (
            patch("paramem.adapters.manifest._hash_safetensors_files") as mock_hash,
            patch("paramem.adapters.manifest._resolve_base_safetensors") as mock_resolve,
        ):
            m = build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                adapter_root=tmp_path,
            )

        assert m.base_model.hash == "sha256:from_readback"
        mock_hash.assert_not_called()
        mock_resolve.assert_not_called()

    def test_readback_invalidates_on_repo_change(self, tmp_path: Path) -> None:
        """Slot with repo='A' does not match model with _name_or_path='B'."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/OTHER", "abc123", "sha256:from_readback"),
        )

        model = self._make_model(name="hf/base", commit="abc123")

        with patch(
            "paramem.adapters.manifest._resolve_base_safetensors", return_value=None
        ) as mock_resolve:
            m = build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                adapter_root=tmp_path,
            )

        # Read-back missed (wrong repo), fell through to file-hash which also returned None
        assert m.base_model.hash == UNKNOWN
        mock_resolve.assert_called_once()

    def test_unknown_commit_sha_skips_readback_but_allows_filehash(self, tmp_path: Path) -> None:
        """Model with _commit_hash=None: read-back skipped, file-hash attempted."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/base", "abc123", "sha256:from_readback"),
        )

        # None commit hash → normalised to UNKNOWN in build_manifest_for
        model = self._make_model(name="hf/base", commit=None)
        model.config._commit_hash = None

        with (
            patch(
                "paramem.adapters.manifest._resolve_base_safetensors", return_value=None
            ) as mock_resolve,
            patch("paramem.adapters.manifest._lookup_hash_from_manifests") as mock_lookup,
        ):
            m = build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                adapter_root=tmp_path,
            )

        # Read-back skipped (UNKNOWN sha)
        mock_lookup.assert_not_called()
        # File-hash attempted
        mock_resolve.assert_called_once()
        # No sources available → UNKNOWN
        assert m.base_model.hash == UNKNOWN

    def test_unknown_slot_hash_is_not_cached(self, tmp_path: Path) -> None:
        """A slot whose base_model.hash == UNKNOWN is excluded; file-hash runs instead."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/base", "abc123", UNKNOWN),
        )

        model = self._make_model(name="hf/base", commit="abc123")

        with patch(
            "paramem.adapters.manifest._resolve_base_safetensors", return_value=None
        ) as mock_resolve:
            m = build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                adapter_root=tmp_path,
            )

        # UNKNOWN slot ignored → file-hash attempted
        mock_resolve.assert_called_once()
        # No sources found → UNKNOWN
        assert m.base_model.hash == UNKNOWN

    def test_no_sources_no_slots_yields_unknown(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty adapter_root + HF cache miss → base_hash UNKNOWN + one warning."""
        import logging

        model = self._make_model(name="hf/bogus_repo_xyz", commit="deadbeef")

        # Attach handler directly — caplog.at_level alone won't capture in this project
        named_logger = logging.getLogger("paramem.adapters.manifest")
        named_logger.addHandler(caplog.handler)
        try:
            with patch("paramem.adapters.manifest._resolve_base_safetensors", return_value=None):
                m = build_manifest_for(
                    model,
                    self._make_tokenizer(),
                    "episodic",
                    registry_path=None,
                    keyed_pairs_path=None,
                    adapter_root=tmp_path,
                )
        finally:
            named_logger.removeHandler(caplog.handler)

        assert m.base_model.hash == UNKNOWN
        warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warn_msgs) >= 1, f"Expected at least one warning; got: {warn_msgs}"

    def test_in_memory_cache_populated_by_readback(self, tmp_path: Path) -> None:
        """The in-memory cache must be populated by the read-back path, not just file-hash."""
        _write_meta(
            tmp_path,
            "episodic",
            "20260421-000000",
            _make_manifest_with_base("hf/base", "abc123", "sha256:readback_hash"),
        )

        model = self._make_model(name="hf/base", commit="abc123")
        cache: dict = {}

        with (
            patch("paramem.adapters.manifest._hash_safetensors_files") as mock_hash,
            patch("paramem.adapters.manifest._resolve_base_safetensors") as mock_resolve,
        ):
            m = build_manifest_for(
                model,
                self._make_tokenizer(),
                "episodic",
                registry_path=None,
                keyed_pairs_path=None,
                base_model_hash_cache=cache,
                adapter_root=tmp_path,
            )

        assert m.base_model.hash == "sha256:readback_hash"
        assert cache[id(model)] == "sha256:readback_hash"
        mock_hash.assert_not_called()
        mock_resolve.assert_not_called()

    def test_multi_shard_order_uses_weight_map_not_filename_sort(self, tmp_path: Path) -> None:
        """_resolve_base_safetensors: shard order comes from weight_map, not filename sort.

        Fixture: two safetensors files where filename-sort order (a < b) differs
        from weight_map order (b listed before a). The hash must match weight_map order.
        """
        import json as _json

        import safetensors.torch
        import torch

        repo_dir = tmp_path / "model_repo"
        repo_dir.mkdir()

        # Create two shard files with different content so order distinguishes them
        f_a = repo_dir / "a.safetensors"
        f_b = repo_dir / "b.safetensors"
        safetensors.torch.save_file({"weight_a": torch.ones(4)}, str(f_a))
        safetensors.torch.save_file({"weight_b": torch.zeros(4)}, str(f_b))

        # weight_map: b listed first, then a (opposite of filename-sort)
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "model.layer.0.weight": "b.safetensors",
                "model.layer.1.weight": "a.safetensors",
            },
        }
        (repo_dir / "model.safetensors.index.json").write_text(_json.dumps(index))

        # _resolve_base_safetensors for a local dir returns sorted(*.safetensors)
        # which would be [a, b]. But that path does NOT read the index.
        # We test the HF-hub code path by patching try_to_load_from_cache.
        from huggingface_hub.file_download import _CACHED_NO_EXIST

        def _fake_cache(repo_id, filename, revision=None):
            mapping = {
                "model.safetensors.index.json": str(repo_dir / "model.safetensors.index.json"),
                "b.safetensors": str(f_b),
                "a.safetensors": str(f_a),
            }
            return mapping.get(filename, _CACHED_NO_EXIST)

        with patch("huggingface_hub.try_to_load_from_cache", side_effect=_fake_cache):
            # Use a fake non-local repo ID so the local-dir branch is skipped
            paths = _resolve_base_safetensors("some_org/some_model", "rev1")

        # weight_map order: b first, then a
        assert paths is not None
        assert paths == [f_b, f_a], f"Expected [b, a] from weight_map order but got {paths}"

        # Confirm hash matches b-then-a concatenation
        expected_hash = "sha256:" + hashlib.sha256(f_b.read_bytes() + f_a.read_bytes()).hexdigest()
        actual_hash = _hash_safetensors_files(paths)
        assert actual_hash == expected_hash
