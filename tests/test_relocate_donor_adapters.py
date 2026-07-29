"""Tests for the one-shot donor relocation migration.

The migration runs once against a live tree holding two donor checkpoints
that cost 37 and 45 minutes of GPU to rebuild, so the round trip it has to
guarantee is exact: after relocation the donor must satisfy
``donor_checkpoint_valid`` — the same gate the seeding hook applies — with
no retraining and no invented fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paramem.training.donor import (
    DONOR_META_FILENAME,
    DONOR_MIN_ENTRIES,
    DONOR_RECIPE_ID,
    _triples_hash,
    donor_checkpoint_valid,
    donor_entries,
    donor_store_dir,
    iter_donor_stores,
)
from scripts.migrate.relocate_donor_adapters import relocate

_BASE_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
_BASE_SHA = "c170c708c41dac9275d15a8fff4eca08d52bab71"
_BASE_HASH = "sha256:f4b6b754b20f151580000000000000000000000000000000000000000000000a"
_LORA_SHAPE = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}
_WEIGHTS = b"donor-weight-bytes"


def _write_tier_slot(adapter_dir: Path, tier: str = "episodic") -> None:
    """Write one main-tier slot manifest — the fingerprint donor source."""
    slot = adapter_dir / tier / "20260728-070421"
    slot.mkdir(parents=True)
    (slot / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "name": tier,
                "trained_at": "2026-07-28T07:04:21Z",
                "base_model": {"repo": _BASE_REPO, "sha": _BASE_SHA, "hash": _BASE_HASH},
                "tokenizer": {
                    "name_or_path": _BASE_REPO,
                    "vocab_size": 32768,
                    "merges_hash": "e553af6fff7d7ad7",
                },
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
                },
                "registry_sha256": "1ec50f93bfa7c",
                "key_count": 210,
                "synthesized": False,
                "window_stamp": "",
            }
        )
    )


def _write_legacy_donor(
    adapter_dir: Path,
    *,
    topology_id: str = "r8-a16-4mod-003cf56e",
    stamp: str = "20260728-051959",
    base_repo: str = _BASE_REPO,
) -> Path:
    """Write a donor slot in the pre-migration ``_donor/<topology>/`` layout."""
    slot = adapter_dir / "_donor" / topology_id / stamp
    slot.mkdir(parents=True)
    (slot / "adapter_model.safetensors").write_bytes(_WEIGHTS)
    (slot / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
        )
    )
    entries = donor_entries(42, DONOR_MIN_ENTRIES)
    (slot / DONOR_META_FILENAME).write_text(
        json.dumps(
            {
                "seed": 42,
                "recipe": DONOR_RECIPE_ID,
                "n_requested": DONOR_MIN_ENTRIES,
                "triples": entries,
                "triples_hash": _triples_hash(entries),
                "weights_sha256": hashlib.sha256(_WEIGHTS).hexdigest(),
                "base_model_id": base_repo,
                "lora_shape": _LORA_SHAPE,
            }
        )
    )
    return slot


class TestRelocate:
    def test_relocated_donor_passes_the_production_validity_gate(self, tmp_path):
        """The whole point: no retrain, and the seeding hook accepts it."""
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)

        moved, skipped = relocate(tmp_path)

        assert (moved, skipped) == (1, 0)
        store = donor_store_dir(tmp_path, _BASE_REPO, _LORA_SHAPE)
        assert donor_checkpoint_valid(store, _BASE_REPO, _LORA_SHAPE) is True

    def test_weights_are_moved_byte_identical(self, tmp_path):
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)

        relocate(tmp_path)

        store = donor_store_dir(tmp_path, _BASE_REPO, _LORA_SHAPE)
        slot = store / "20260728-051959"
        assert slot.joinpath("adapter_model.safetensors").read_bytes() == _WEIGHTS

    def test_manifest_carries_the_real_fingerprint_not_unknown(self, tmp_path):
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)

        relocate(tmp_path)

        store = donor_store_dir(tmp_path, _BASE_REPO, _LORA_SHAPE)
        manifest = json.loads((store / "20260728-051959" / "meta.json").read_text())
        assert manifest["base_model"] == {
            "repo": _BASE_REPO,
            "sha": _BASE_SHA,
            "hash": _BASE_HASH,
        }
        assert manifest["tokenizer"]["vocab_size"] == 32768
        assert manifest["synthesized"] is False
        # Empty registry hash is what makes find_live_slot resolve a donor slot.
        assert manifest["registry_sha256"] == ""
        assert manifest["key_count"] == len(donor_entries(42, DONOR_MIN_ENTRIES))

    def test_donor_meta_drops_the_fields_the_manifest_now_owns(self, tmp_path):
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)

        relocate(tmp_path)

        store = donor_store_dir(tmp_path, _BASE_REPO, _LORA_SHAPE)
        meta = json.loads((store / "20260728-051959" / DONOR_META_FILENAME).read_text())
        assert "base_model_id" not in meta
        assert "lora_shape" not in meta
        assert meta["recipe"] == DONOR_RECIPE_ID

    def test_two_topologies_land_in_separate_stores(self, tmp_path):
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)
        proc_shape = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }
        proc_slot = _write_legacy_donor(
            tmp_path, topology_id="r8-a16-7mod-b102f771", stamp="20260728-065524"
        )
        cfg = json.loads((proc_slot / "adapter_config.json").read_text())
        cfg["target_modules"] = proc_shape["target_modules"]
        (proc_slot / "adapter_config.json").write_text(json.dumps(cfg))
        meta = json.loads((proc_slot / DONOR_META_FILENAME).read_text())
        meta["lora_shape"] = proc_shape
        (proc_slot / DONOR_META_FILENAME).write_text(json.dumps(meta))

        moved, skipped = relocate(tmp_path)

        assert (moved, skipped) == (2, 0)
        assert len(iter_donor_stores(tmp_path)) == 2

    def test_legacy_tree_is_removed_when_emptied(self, tmp_path):
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)

        relocate(tmp_path)

        assert not (tmp_path / "_donor").exists()

    def test_rerun_is_a_no_op(self, tmp_path):
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)

        relocate(tmp_path)
        moved, skipped = relocate(tmp_path)

        assert (moved, skipped) == (0, 0)
        store = donor_store_dir(tmp_path, _BASE_REPO, _LORA_SHAPE)
        assert donor_checkpoint_valid(store, _BASE_REPO, _LORA_SHAPE) is True

    def test_donor_without_a_matching_tier_manifest_is_left_in_place(self, tmp_path):
        """Never invent a fingerprint: report and leave the slot untouched."""
        _write_tier_slot(tmp_path)
        slot = _write_legacy_donor(tmp_path, base_repo="some/other-base")

        moved, skipped = relocate(tmp_path)

        assert (moved, skipped) == (0, 1)
        assert slot.exists()
        assert (slot / "adapter_model.safetensors").read_bytes() == _WEIGHTS

    def test_dry_run_touches_nothing(self, tmp_path):
        _write_tier_slot(tmp_path)
        slot = _write_legacy_donor(tmp_path)

        moved, skipped = relocate(tmp_path, dry_run=True)

        assert (moved, skipped) == (1, 0)
        assert slot.exists()
        assert not (slot / "meta.json").exists()
        assert iter_donor_stores(tmp_path) == []

    def test_no_legacy_tree_is_a_clean_no_op(self, tmp_path):
        _write_tier_slot(tmp_path)
        assert relocate(tmp_path) == (0, 0)

    def test_missing_adapter_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            relocate(tmp_path / "absent")

    def test_legacy_tree_removed_when_only_scaffolding_remains(self, tmp_path):
        """Training scratch is regenerable; a completed relocation leaves no
        dead _donor/ tree behind because of it."""
        _write_tier_slot(tmp_path)
        slot = _write_legacy_donor(tmp_path)
        scratch = slot.parent / ".training_scratch"
        scratch.mkdir()
        (scratch / "epoch_log.json").write_text("[]")
        (slot.parent / ".pending").mkdir()

        relocate(tmp_path)

        assert not (tmp_path / "_donor").exists()

    def test_legacy_tree_kept_when_a_checkpoint_did_not_relocate(self, tmp_path):
        """Never rmtree over an unrelocated checkpoint."""
        _write_tier_slot(tmp_path)
        _write_legacy_donor(tmp_path)
        stranded = _write_legacy_donor(
            tmp_path,
            topology_id="r8-a16-9mod-deadbeef",
            stamp="20260101-000000",
            base_repo="some/other-base",
        )

        moved, skipped = relocate(tmp_path)

        assert (moved, skipped) == (1, 1)
        assert stranded.exists()
        assert (tmp_path / "_donor").exists()
