"""Tests for paramem.adapters.slot -- the one promotion envelope shared by
both venues (train weight payloads and simulate graph payloads).

Covers the slot envelope's design: one promotion sequence for both venues,
a payload digest computed fresh from the bytes just written (never passed
through from the caller), and every manifest-bearing save path ending with
a manifest whose ``payload.sha256`` equals the sha256 of the payload bytes
actually on disk -- pinned here for the weight write
(``atomic_save_adapter``) and the per-tier commit primitive's train arm
(``commit_tier_slot``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from paramem.adapters.manifest import graph_payload_manifest, iter_slot_candidates, read_manifest
from paramem.adapters.slot import required_slot_files, write_slot
from tests._manifest_fixtures import make_train_manifest


def _tree_snapshot(root: Path) -> dict:
    """Map every file under *root* to its bytes, for an exact before/after
    content comparison -- catches a cleanup that corrupts or truncates a
    survivor, not just one that deletes it outright."""
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class TestWriteSlot:
    def test_write_writes_payload_and_manifest_into_one_promoted_slot(self, tmp_path):
        tier_root = tmp_path / "episodic"
        manifest = make_train_manifest(name="episodic")

        def _write(pending):
            (pending / "adapter_model.safetensors").write_bytes(b"fake-weights")
            (pending / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')

        slot = write_slot(tier_root, manifest=manifest, write_payload=_write)

        assert slot.parent == tier_root
        assert ".pending" not in slot.parts
        assert (slot / "meta.json").exists()
        assert (slot / "adapter_model.safetensors").exists()
        assert (slot / "adapter_config.json").exists()
        assert list(iter_slot_candidates(tier_root)) == [slot]
        # The staging dir was promoted by rename -- nothing left pending.
        assert not (tier_root / ".pending" / slot.name).exists()

    def test_write_stamps_the_digest_of_the_bytes_it_just_wrote(self, tmp_path):
        tier_root = tmp_path / "episodic"
        manifest = make_train_manifest(name="episodic")
        payload_bytes = b"fake-weights-for-digest-check"

        def _write(pending):
            (pending / "adapter_model.safetensors").write_bytes(payload_bytes)
            (pending / "adapter_config.json").write_bytes(b"{}")

        slot = write_slot(tier_root, manifest=manifest, write_payload=_write)

        stamped = read_manifest(slot)
        assert stamped.payload.sha256 == hashlib.sha256(payload_bytes).hexdigest()
        # The manifest handed to write_slot carried no digest -- it was
        # computed fresh from the bytes just written, not passed through.
        assert manifest.payload.sha256 == ""

    def test_write_leaves_no_slot_when_the_payload_write_raises(self, tmp_path):
        tier_root = tmp_path / "episodic"
        manifest = make_train_manifest(name="episodic")

        def _write(pending):
            (pending / "adapter_model.safetensors").write_bytes(b"partial")
            raise RuntimeError("simulated write failure")

        with pytest.raises(RuntimeError, match="simulated write failure"):
            write_slot(tier_root, manifest=manifest, write_payload=_write)

        assert list(iter_slot_candidates(tier_root)) == [], (
            "a failed payload write must never leave a promoted (non-pending) slot behind"
        )

    def test_both_venues_produce_the_same_slot_shape(self, tmp_path):
        train_root = tmp_path / "episodic"
        simulate_root = tmp_path / "semantic"

        train_manifest = make_train_manifest(name="episodic")

        def _write_train(pending):
            (pending / "adapter_model.safetensors").write_bytes(b"fake-weights")
            (pending / "adapter_config.json").write_bytes(b"{}")

        train_slot = write_slot(train_root, manifest=train_manifest, write_payload=_write_train)

        simulate_manifest = graph_payload_manifest(
            name="semantic", key_count=2, registry_sha256="", window_stamp=""
        )

        def _write_simulate(pending):
            (pending / "graph.json").write_bytes(b'{"nodes": []}')

        simulate_slot = write_slot(
            simulate_root, manifest=simulate_manifest, write_payload=_write_simulate
        )

        for slot, kind, root in (
            (train_slot, "train", train_root),
            (simulate_slot, "simulate", simulate_root),
        ):
            assert slot.parent == root
            assert ".pending" not in slot.parts
            names = {p.name for p in slot.iterdir()}
            assert names == set(required_slot_files(kind)), (
                f"{kind} slot must carry exactly its required_slot_files set"
            )
            manifest = read_manifest(slot)
            assert manifest.payload.kind == kind


class _FakeAtomicSaveModel:
    """Minimal ``save_pretrained`` stand-in for :func:`atomic_save_adapter` --
    no PEFT/torch dependency, just a real filesystem write so the digest
    computed by ``write_slot`` is over real bytes."""

    def __init__(self, weight_bytes: bytes) -> None:
        self._weight_bytes = weight_bytes

    def save_pretrained(self, path, selected_adapters=None) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(self._weight_bytes)


class TestPayloadDigestAcrossManifestBearingSavePaths:
    """``test_every_manifest_bearing_save_path_stamps_a_payload_digest`` --
    the promoted slot's manifest carries a ``payload.sha256`` equal to the
    sha256 of the payload bytes on disk, pinned independently at each save
    path that hands a payload to ``write_slot``."""

    def test_atomic_save_adapter_stamps_the_digest_of_the_payload_bytes_on_disk(self, tmp_path):
        from paramem.models.loader import atomic_save_adapter

        weight_bytes = b"atomic-save-adapter-payload-bytes"
        model = _FakeAtomicSaveModel(weight_bytes)
        manifest = make_train_manifest(name="episodic")

        slot = atomic_save_adapter(model, tmp_path / "episodic", "episodic", manifest=manifest)

        stamped = read_manifest(slot)
        on_disk = (slot / "adapter_model.safetensors").read_bytes()
        assert stamped.payload.sha256 == hashlib.sha256(on_disk).hexdigest()

    def test_commit_tier_slot_train_arm_stamps_the_digest_of_the_payload_bytes_on_disk(
        self, tmp_path
    ):
        from paramem.memory.persistence import commit_tier_slot
        from tests._fold_fixtures import _FakeModel, _make_loop

        loop = _make_loop(tmp_path)
        loop.model = _FakeModel()

        written = commit_tier_slot(
            loop=loop,
            tier="episodic",
            adapter_name="episodic",
            stamp="20260101T0000",
            mode="train",
            all_keyed=[],
            output_dir=loop.output_dir,
        )

        assert written is not None
        stamped = read_manifest(written)
        on_disk = (written / "adapter_model.safetensors").read_bytes()
        assert stamped.payload.sha256 == hashlib.sha256(on_disk).hexdigest()


class TestCommitTierSlotCrashCleanup:
    """``commit_tier_slot``'s orphan-cleanup ``finally`` (``paramem.memory.persistence``)
    removes only the ONE timestamped slot dir the failing call itself wrote --
    never ``slot_root``, which also holds prior committed slots, the tier
    registry file, and (for episodic) the ``interim_*`` siblings. Pinned here
    across the three crash-window regimes (pre-bind, mid-window, post-flush)
    and both binding sites (``save_adapter`` for train, ``write_slot`` for
    simulate)."""

    def test_train_arm_mid_window_failure_deletes_only_the_torn_slot(self, tmp_path, monkeypatch):
        from paramem.memory.interim_adapter import interim_dir_for_name
        from paramem.memory.persistence import commit_tier_slot
        from tests._fold_fixtures import _FakeModel, _make_loop

        loop = _make_loop(tmp_path)
        loop.model = _FakeModel()

        slot_a = commit_tier_slot(
            loop=loop,
            tier="episodic",
            adapter_name="episodic",
            stamp="20260101T0000",
            mode="train",
            all_keyed=[],
            output_dir=loop.output_dir,
        )
        tier_root = loop.output_dir / "episodic"
        registry_path = tier_root / "indexed_key_registry.json"

        # A sibling interim ring member -- lives directly under the same
        # tier root as the main-tier slots and must survive untouched.
        interim_dir = interim_dir_for_name(loop.output_dir, "episodic_interim_20260101T0000")
        interim_dir.mkdir(parents=True)
        (interim_dir / "sentinel.json").write_text('{"sentinel": true}')

        slot_a_before = _tree_snapshot(slot_a)
        registry_before = registry_path.read_bytes()
        interim_before = _tree_snapshot(interim_dir)
        slots_before = set(iter_slot_candidates(tier_root))

        import paramem.memory.persistence as persistence_mod

        def _raise_after_slot_write(*args, **kwargs):
            raise RuntimeError("simulated key-metadata write failure")

        monkeypatch.setattr(persistence_mod, "_write_tier_key_metadata", _raise_after_slot_write)

        with pytest.raises(RuntimeError, match="simulated key-metadata write failure"):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic",
                stamp="20260101T0001",
                mode="train",
                all_keyed=[],
                output_dir=loop.output_dir,
            )

        assert set(iter_slot_candidates(tier_root)) == slots_before, (
            "the torn call's own slot dir must be gone, and nothing else must appear"
        )
        assert _tree_snapshot(slot_a) == slot_a_before, "slot A must survive byte-for-byte"
        assert registry_path.read_bytes() == registry_before, "the tier registry must be untouched"
        assert _tree_snapshot(interim_dir) == interim_before, (
            "the interim sibling must never be touched by a main-tier slot's cleanup"
        )

    def test_train_arm_pre_bind_failure_deletes_nothing_under_the_tier_root(
        self, tmp_path, monkeypatch
    ):
        from paramem.memory.persistence import commit_tier_slot
        from tests._fold_fixtures import _FakeModel, _make_loop

        loop = _make_loop(tmp_path)
        loop.model = _FakeModel()

        slot_a = commit_tier_slot(
            loop=loop,
            tier="episodic",
            adapter_name="episodic",
            stamp="20260101T0000",
            mode="train",
            all_keyed=[],
            output_dir=loop.output_dir,
        )
        assert slot_a is not None
        tier_root = loop.output_dir / "episodic"
        before = _tree_snapshot(tier_root)

        import paramem.models.loader as loader_mod

        def _raise_before_bind(*args, **kwargs):
            raise RuntimeError("simulated save_adapter failure")

        monkeypatch.setattr(loader_mod, "save_adapter", _raise_before_bind)

        with pytest.raises(RuntimeError, match="simulated save_adapter failure"):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic",
                stamp="20260101T0001",
                mode="train",
                all_keyed=[],
                output_dir=loop.output_dir,
            )

        assert _tree_snapshot(tier_root) == before, (
            "a failure before the slot write returns has nothing bound to remove -- "
            "the finally block must delete nothing"
        )

    def test_train_arm_post_flush_failure_in_prune_keeps_the_newly_committed_slot_live(
        self, tmp_path, monkeypatch
    ):
        from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
        from paramem.memory.persistence import commit_tier_slot
        from tests._fold_fixtures import _FakeModel, _make_loop

        loop = _make_loop(tmp_path)
        loop.model = _FakeModel()

        slot_a = commit_tier_slot(
            loop=loop,
            tier="episodic",
            adapter_name="episodic",
            stamp="20260101T0000",
            mode="train",
            all_keyed=[],
            output_dir=loop.output_dir,
        )
        tier_root = loop.output_dir / "episodic"
        slots_before = set(iter_slot_candidates(tier_root))

        import paramem.memory.persistence as persistence_mod

        def _raise_in_prune(*args, **kwargs):
            raise RuntimeError("simulated prune failure")

        monkeypatch.setattr(persistence_mod, "prune_old_slots", _raise_in_prune)

        with pytest.raises(RuntimeError, match="simulated prune failure"):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic",
                stamp="20260101T0001",
                mode="train",
                all_keyed=[],
                output_dir=loop.output_dir,
            )

        slots_after = set(iter_slot_candidates(tier_root))
        new_slots = slots_after - slots_before
        assert len(new_slots) == 1, (
            "the newly committed slot must survive -- prune raising is after the commit signal"
        )
        new_slot = new_slots.pop()
        assert slot_a in slots_after, "the prior slot is untouched by a prune failure"

        live = find_live_slot(tier_root, tier_registry_sha256(tier_root))
        assert live == new_slot, "the registry must now point at the newly committed slot"

    def test_simulate_arm_mid_window_failure_deletes_only_the_torn_slot(
        self, tmp_path, monkeypatch
    ):
        from paramem.memory.interim_adapter import interim_dir_for_name
        from paramem.memory.persistence import commit_tier_slot
        from tests._fold_fixtures import _make_loop

        loop = _make_loop(tmp_path)

        slot_a = commit_tier_slot(
            loop=loop,
            tier="episodic",
            adapter_name="episodic",
            stamp="20260101T0000",
            mode="simulate",
            all_keyed=[],
            output_dir=loop.output_dir,
        )
        tier_root = loop.output_dir / "episodic"
        registry_path = tier_root / "indexed_key_registry.json"

        interim_dir = interim_dir_for_name(loop.output_dir, "episodic_interim_20260101T0000")
        interim_dir.mkdir(parents=True)
        (interim_dir / "sentinel.json").write_text('{"sentinel": true}')

        slot_a_before = _tree_snapshot(slot_a)
        registry_before = registry_path.read_bytes()
        interim_before = _tree_snapshot(interim_dir)
        slots_before = set(iter_slot_candidates(tier_root))

        import paramem.memory.persistence as persistence_mod

        def _raise_after_slot_write(*args, **kwargs):
            raise RuntimeError("simulated key-metadata write failure")

        monkeypatch.setattr(persistence_mod, "_write_tier_key_metadata", _raise_after_slot_write)

        with pytest.raises(RuntimeError, match="simulated key-metadata write failure"):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic",
                stamp="20260101T0001",
                mode="simulate",
                all_keyed=[],
                output_dir=loop.output_dir,
            )

        assert set(iter_slot_candidates(tier_root)) == slots_before, (
            "the torn simulate-arm slot (bound via write_slot) must be gone"
        )
        assert _tree_snapshot(slot_a) == slot_a_before, "slot A must survive byte-for-byte"
        assert registry_path.read_bytes() == registry_before, "the tier registry must be untouched"
        assert _tree_snapshot(interim_dir) == interim_before, (
            "the interim sibling must never be touched by a main-tier slot's cleanup"
        )
