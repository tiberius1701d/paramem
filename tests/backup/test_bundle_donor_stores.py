"""Donor stores round-trip through the shared bundle write/restore path.

A donor is an adapter store, so it is captured and restored by the same
routines that own the tier stores — no donor-specific backup code. What
this pins is the consequence: before donors became stores, ``write_bundle``
never captured them and ``restore_bundle``'s stray sweep ``rmtree``'d them
unconditionally, so any restore or migration rollback destroyed 37-45
minutes of GPU per store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paramem.backup.backup import restore_bundle, write_bundle
from paramem.training.donor import (
    DONOR_META_FILENAME,
    DONOR_MIN_ENTRIES,
    DONOR_RECIPE_ID,
    donor_checkpoint_valid,
    donor_entries,
    donor_store_dir,
    iter_donor_stores,
    triples_hash,
)
from tests.backup.test_restore import _make_source_store

_BASE_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
_LORA_SHAPE = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
_DONOR_WEIGHTS = b"donor-weights-that-cost-45-minutes"


def _seed_donor_store(adapters_root: Path) -> Path:
    """Write a valid donor store under *adapters_root*; return the store dir."""
    store = donor_store_dir(adapters_root, _BASE_REPO, _LORA_SHAPE)
    slot = store / "20260728-051959"
    slot.mkdir(parents=True)
    (slot / "adapter_model.safetensors").write_bytes(_DONOR_WEIGHTS)
    (slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')
    (slot / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "name": store.name,
                "trained_at": "2026-07-28T05:19:59Z",
                "window_stamp": "",
                "base_model": {
                    "repo": _BASE_REPO,
                    "sha": "c170c708abc",
                    "hash": "sha256:f4b6abc",
                },
                "tokenizer": {
                    "name_or_path": _BASE_REPO,
                    "vocab_size": 32000,
                    "merges_hash": "d" * 64,
                },
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                },
                # Empty: a donor carries no key registry. This is what makes
                # find_live_slot resolve it.
                "registry_sha256": "",
                "key_count": 147,
                "synthesized": False,
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
                "triples_hash": triples_hash(entries),
                "weights_sha256": hashlib.sha256(_DONOR_WEIGHTS).hexdigest(),
            }
        )
    )
    return store


def _bundle_with_donor(tmp_path: Path) -> tuple[Path, dict, Path]:
    src = _make_source_store(tmp_path)
    adapters_root = next(iter(src["adapter_dirs"].values())).parent
    store = _seed_donor_store(adapters_root)
    slot = write_bundle(
        config_path=src["config_path"],
        registry_path=src["registry_path"],
        adapter_dirs=src["adapter_dirs"],
        base_dir=src["bundle_base"],
        meta_fields={"tier": "manual", "label": "donor-test"},
        adapter_scope="live",
        speaker_profiles_path=src["speaker_profiles_path"],
    )
    return slot, src, store


class TestDonorStoreInBundle:
    def test_write_bundle_captures_the_donor_store(self, tmp_path) -> None:
        bundle_slot, _src, store = _bundle_with_donor(tmp_path)

        manifest = json.loads((bundle_slot / "bundle.meta.json").read_text())
        paths = {entry["path"] for entry in manifest["files"]}

        assert f"adapters/{store.name}/adapter_model.safetensors" in paths
        # donor_meta.json must ride along: without it the restored donor
        # fails donor_checkpoint_valid and is rebuilt at full GPU cost.
        assert f"adapters/{store.name}/{DONOR_META_FILENAME}" in paths

    def test_restore_puts_the_donor_back_valid(self, tmp_path) -> None:
        """The round trip that matters: after restore, the donor satisfies the
        same gate the seeding hook applies."""
        bundle_slot, _src, store = _bundle_with_donor(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=scratch / "server.yaml",
            restore_config=False,
        )

        restored = scratch / "adapters" / store.name
        assert restored.is_dir()
        assert donor_checkpoint_valid(restored, _BASE_REPO, _LORA_SHAPE) is True

    def test_restored_donor_weights_are_byte_identical(self, tmp_path) -> None:
        bundle_slot, _src, store = _bundle_with_donor(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=scratch / "server.yaml",
            restore_config=False,
        )

        restored_stores = iter_donor_stores(scratch / "adapters")
        assert len(restored_stores) == 1
        _name, restored = restored_stores[0]
        slots = [p for p in restored.iterdir() if p.is_dir()]
        assert len(slots) == 1
        assert slots[0].joinpath("adapter_model.safetensors").read_bytes() == _DONOR_WEIGHTS

    def test_restore_does_not_delete_a_donor_it_just_wrote(self, tmp_path) -> None:
        """The stray sweep used to rmtree every unrecognised top-level entry,
        which would remove the store the restore had just written."""
        bundle_slot, _src, store = _bundle_with_donor(tmp_path)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        # Pre-existing donor store in the target, same name as the bundle's.
        _seed_donor_store(scratch / "adapters")

        restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=scratch / "server.yaml",
            restore_config=False,
        )

        assert (scratch / "adapters" / store.name).is_dir()
        assert (
            donor_checkpoint_valid(scratch / "adapters" / store.name, _BASE_REPO, _LORA_SHAPE)
            is True
        )

    def test_donor_store_absent_from_bundle_is_pruned_as_an_orphan(self, tmp_path) -> None:
        """Complete pruning: a store the recovery set does not contain is not
        part of the restored state, and the safety bundle written first can
        bring it back. Reported, never silent."""
        src = _make_source_store(tmp_path)
        bundle_slot = write_bundle(
            config_path=src["config_path"],
            registry_path=src["registry_path"],
            adapter_dirs=src["adapter_dirs"],
            base_dir=src["bundle_base"],
            meta_fields={"tier": "manual", "label": "no-donor"},
            adapter_scope="live",
            speaker_profiles_path=src["speaker_profiles_path"],
        )
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        orphan = _seed_donor_store(scratch / "adapters")

        result = restore_bundle(
            bundle_slot,
            data_dir=scratch,
            config_path=scratch / "server.yaml",
            restore_config=False,
        )

        assert not orphan.exists()
        pruned = {(p["name"], p["kind"]) for p in result.pruned_orphans}
        assert (orphan.name, "donor") in pruned
