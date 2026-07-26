"""Tests for the donor-adapter lifecycle (paramem.training.donor).

CPU-only. No GPU, no real model weights -- ``build_donor``/the seeding
hook are exercised against mocked PEFT primitives and a bare
``ConsolidationLoop`` built via ``object.__new__`` (same pattern
``tests/test_consolidation.py`` already uses for funnel-level unit tests),
never a real ``from_pretrained`` load.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.donor import (
    _FIXTURE_PATH,
    DONOR_BUILD_ADAPTER_NAME,
    DONOR_KEY_BAND_WIDTH,
    DONOR_LOAD_ADAPTER_NAME,
    DONOR_MIN_ENTRIES,
    DonorBuildIncomplete,
    _latest_donor_slot,
    build_donor,
    donor_checkpoint_dir,
    donor_checkpoint_valid,
    donor_entries,
    load_donor_into_transient_slot,
)
from paramem.utils.config import AdapterConfig, TrainingConfig

# Computed once at fixture-authoring time. Fixture content is fully
# fictional -- every subject/object value naming a real person, place,
# organisation, or date was verified disjoint from any real capture at
# authoring time (word-token intersection, minus stopwords/predicates, is
# empty); this test only guards against silent, unnoticed drift of the
# tracked fixture file itself afterward.
_FIXTURE_SHA256 = "0c748e61af268ba1b13eee0c01752e4dea2327be67353a2ff30d540b44f959e1"

_SINGLE_VALUED_PREDICATES = ("birth date", "graduation date", "has spouse")

_LORA_SHAPE = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}


def _make_bare_loop(tmp_path: Path) -> ConsolidationLoop:
    """Minimal ConsolidationLoop stub with only the attributes
    ``_train_tier_adapter`` / ``_maybe_seed_from_donor`` read.

    Bypasses ``__init__`` (``object.__new__``) to avoid any model/GPU
    requirement -- the same pattern
    ``tests/test_consolidation.py::TestRunConsolidationCycleAbortPath._make_minimal_loop``
    and ``tests/test_run_consolidation_cycle.py::_make_mock_loop`` already use.
    """
    from peft import PeftModel

    loop = object.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.model.__class__ = PeftModel
    loop.model.peft_config = {"episodic": MagicMock(), "procedural": MagicMock()}
    # Real dict mutation on delete -- makes "transient slot deleted/not
    # deleted" assertions meaningful (M1 fix: the prior fake create_adapter
    # never registered the transient name, so with a MagicMock
    # delete_adapter that does nothing either, those assertions passed
    # whether or not the cleanup code ran at all).
    loop.model.delete_adapter.side_effect = lambda name: loop.model.peft_config.pop(name, None)
    loop.model.get_base_model.return_value.config._name_or_path = "test/base-model"
    loop.tokenizer = MagicMock()
    loop.training_config = TrainingConfig()
    loop.episodic_config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop._thermal_policy = None
    return loop


def _write_donor_slot(
    checkpoint_dir: Path,
    *,
    stamp: str = "20260101-000000",
    base_model_id: str = "test/base",
    lora_shape: dict | None = None,
    meta_text: str | None = None,
    write_weights: bool = True,
) -> Path:
    """Write a minimal on-disk donor slot for TestDonorCheckpoint fixtures."""
    slot = checkpoint_dir / stamp
    slot.mkdir(parents=True, exist_ok=True)
    if write_weights:
        (slot / "adapter_model.safetensors").write_bytes(b"weights")
    if meta_text is not None:
        (slot / "donor_meta.json").write_text(meta_text)
    else:
        meta = {"base_model_id": base_model_id, "lora_shape": lora_shape or _LORA_SHAPE}
        (slot / "donor_meta.json").write_text(json.dumps(meta))
    return slot


class TestDonorFixture:
    """The tracked fixture (donor_fixture.json) is the ONLY donor artifact
    with real-data provenance -- guard it against silent drift and never
    read the scratchpad source from this suite (test-scoping clause)."""

    def test_fixture_sha256_unchanged(self):
        """Drift guard: any edit to donor_fixture.json must be a deliberate,
        reviewed change -- not a silent structural regression."""
        digest = hashlib.sha256(_FIXTURE_PATH.read_bytes()).hexdigest()
        assert digest == _FIXTURE_SHA256, (
            "donor_fixture.json changed -- if intentional, update _FIXTURE_SHA256 "
            f"to {digest!r} after re-verifying the source<->fixture token "
            "intersection is still empty"
        )

    def test_fixture_keys_verbatim(self):
        """The 21 donor-band keys (the mechanism) must be exactly preserved.

        Remapped from the original captured-live production numerals
        (``graph179-193`` / ``proc35-40``) to ``graph101-115`` /
        ``proc101-106`` after the small-N validation runs proved donor
        seeding transfers with zero key overlap (see donor.py's module
        docstring "Fixture provenance" section) -- these numerals sit
        outside every documented live-store key range by construction.
        """
        with _FIXTURE_PATH.open() as f:
            fixture = json.load(f)
        keys = [e["key"] for e in fixture]
        assert keys == [
            "graph101",
            "graph102",
            "graph103",
            "graph104",
            "graph105",
            "graph106",
            "graph107",
            "graph108",
            "graph109",
            "graph110",
            "graph111",
            "graph112",
            "graph113",
            "graph114",
            "graph115",
            "proc101",
            "proc102",
            "proc103",
            "proc104",
            "proc105",
            "proc106",
        ]

    def test_fixture_keys_outside_live_store_ranges(self):
        """Band-membership guard: the donor's block-0 keys must not fall
        inside any documented live-store range (graph141-148/156-158/
        168-174/179-193, proc<=~42) -- the whole point of the post-
        validation remap was to eliminate overlap-by-construction."""
        with _FIXTURE_PATH.open() as f:
            fixture = json.load(f)
        live_graph_ranges = [(141, 148), (156, 158), (168, 174), (179, 193)]
        live_proc_ceiling = 42
        for entry in fixture:
            key = entry["key"]
            if key.startswith("graph"):
                num = int(key.removeprefix("graph"))
                assert 101 <= num <= 115
                for lo, hi in live_graph_ranges:
                    assert not (lo <= num <= hi), f"{key} collides with live range {lo}-{hi}"
            elif key.startswith("proc"):
                num = int(key.removeprefix("proc"))
                assert 101 <= num <= 106
                assert num > live_proc_ceiling

    def test_fixture_entries_have_required_shape(self):
        with _FIXTURE_PATH.open() as f:
            fixture = json.load(f)
        assert len(fixture) == 21
        for entry in fixture:
            assert set(entry.keys()) == {"key", "subject", "predicate", "object"}

    def test_fixture_pool_disjoint_from_donor_content_pools(self):
        """donor.py's synthetic content pools must never reproduce a fixture
        value verbatim (H1 docstring-accuracy fix): every pool is checked
        against the fixture's own 21 subject/object values."""
        import paramem.training.donor as donor_module

        with _FIXTURE_PATH.open() as f:
            fixture = json.load(f)
        fixture_values = {e["subject"] for e in fixture} | {e["object"] for e in fixture}

        pools = [
            donor_module._PRIMARY_NAMES,
            donor_module._SPOUSE_NAMES,
            donor_module._CHILD_NAMES,
            [x for pair in donor_module._PLACE_PAIRS for x in pair],
            [x for pair in donor_module._PROJECTS for x in pair],
            donor_module._EXPERTISE_PHRASES,
            donor_module._BACKGROUND_PHRASES,
            donor_module._UNIVERSITIES,
            donor_module._COUNTRIES,
            donor_module._HOBBIES,
        ]
        for pool in pools:
            overlap = set(pool) & fixture_values
            assert not overlap, f"pool entry reproduces a fixture value verbatim: {overlap}"


class TestDonorEntries:
    """donor_entries(seed, n) is a pure function: same seed -> identical
    triple set; different seed -> different. Structural properties
    (cluster shape, tier mix, key band, triple/consistency invariants)
    must hold at every valid n and seed."""

    def test_rejects_n_below_floor(self):
        with pytest.raises(ValueError, match="DONOR_MIN_ENTRIES|128"):
            donor_entries(seed=1, n=DONOR_MIN_ENTRIES - 1)

    def test_returns_at_least_n_and_whole_blocks(self):
        entries = donor_entries(seed=1, n=DONOR_MIN_ENTRIES)
        assert len(entries) >= DONOR_MIN_ENTRIES
        assert len(entries) % 21 == 0, "must return whole 21-entry structural blocks"

    def test_determinism_same_seed_identical(self):
        a = donor_entries(seed=7, n=128)
        b = donor_entries(seed=7, n=128)
        assert a == b

    def test_different_seed_differs(self):
        a = donor_entries(seed=7, n=128)
        b = donor_entries(seed=8, n=128)
        assert a != b

    def test_both_prefixes_present(self):
        entries = donor_entries(seed=1, n=128)
        assert any(e["key"].startswith("graph") for e in entries)
        assert any(e["key"].startswith("proc") for e in entries)

    def test_keys_within_reserved_band(self):
        entries = donor_entries(seed=1, n=200)
        for e in entries:
            if e["key"].startswith("graph"):
                assert int(e["key"].removeprefix("graph")) <= DONOR_KEY_BAND_WIDTH
            elif e["key"].startswith("proc"):
                assert int(e["key"].removeprefix("proc")) <= DONOR_KEY_BAND_WIDTH

    def test_keys_unique_no_collisions(self):
        entries = donor_entries(seed=3, n=147)
        keys = [e["key"] for e in entries]
        assert len(keys) == len(set(keys)), "generated blocks must never collide keys"

    def test_crowded_expertise_cluster_shape_preserved_per_block(self):
        """Every 21-entry block must carry the template's crowded 7-wide
        same-(subject,predicate) 'expertise' cluster -- the structural
        property the donor exists to teach (divergence depth). Block 0 is
        the real fixture (subject "speaker0"); every later block has its
        own distinct primary subject (H1 fix) -- detected here as the
        block's own first entry's subject, not hardcoded."""
        entries = donor_entries(seed=5, n=128)
        for start in range(0, len(entries), 21):
            block = entries[start : start + 21]
            primary = block[0]["subject"]
            counts = Counter((e["subject"], e["predicate"]) for e in block)
            assert counts[(primary, "expertise")] == 7
            assert counts[(primary, "background in")] == 2
            assert counts[(primary, "has interest")] == 3

    def test_tier_mix_matches_template_per_block(self):
        entries = donor_entries(seed=5, n=128)
        for start in range(0, len(entries), 21):
            block = entries[start : start + 21]
            graph_count = sum(1 for e in block if e["key"].startswith("graph"))
            proc_count = sum(1 for e in block if e["key"].startswith("proc"))
            assert graph_count == 15
            assert proc_count == 6

    def test_zero_duplicate_triples_at_multiple_seeds(self):
        """H1: no (subject, predicate, object) triple may ever be trained
        under more than one key -- that is exactly the crowded-predicate
        collapse pattern the donor exists to cure."""
        for seed in range(10):
            entries = donor_entries(seed, 128)
            triples = [(e["subject"], e["predicate"], e["object"]) for e in entries]
            assert len(triples) == len(set(triples)), (
                f"seed={seed}: duplicate (subject,predicate,object) triple(s) found "
                f"-- {len(triples) - len(set(triples))} redundant"
            )

    def test_no_conflicting_objects_for_single_valued_predicates_at_multiple_seeds(self):
        """H1: no subject may carry two different objects for a
        single-cardinality predicate (has spouse / graduation date /
        birth date) -- that would be contradictory training data, distinct
        from the intentionally multi-valued clusters (expertise etc.)."""
        for seed in range(10):
            entries = donor_entries(seed, 128)
            seen: dict[tuple[str, str], str] = {}
            for e in entries:
                if e["predicate"] not in _SINGLE_VALUED_PREDICATES:
                    continue
                key = (e["subject"], e["predicate"])
                if key in seen:
                    assert seen[key] == e["object"], (
                        f"seed={seed}: subject {e['subject']!r} has conflicting "
                        f"objects for predicate {e['predicate']!r}: "
                        f"{seen[key]!r} vs {e['object']!r}"
                    )
                else:
                    seen[key] = e["object"]

    def test_large_n_does_not_crash_when_name_pools_exhausted(self):
        """_draw_unique's deterministic-extension fallback must keep
        donor_entries working (no crash, still zero duplicates) past the
        point where the 10-entry spouse/child name pools are exhausted
        (12 synthetic blocks > 10 -- still within the graph band's capacity:
        200 total minus the fixture's own reserved 15 leaves 185 free, and
        12 blocks x 15 keys/block = 180)."""
        entries = donor_entries(seed=11, n=21 * 13)  # 13 blocks -> 12 synthetic
        triples = [(e["subject"], e["predicate"], e["object"]) for e in entries]
        assert len(triples) == len(set(triples))


class TestDonorCheckpoint:
    """Checkpoint persistence + validation primitives."""

    def test_donor_checkpoint_dir_is_a_dedicated_subdir(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        assert ckpt == tmp_path / "_donor"
        # Must not collide with any production tier name.
        assert ckpt.name not in ("episodic", "semantic", "procedural")

    def test_donor_checkpoint_valid_false_when_absent(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        assert donor_checkpoint_valid(ckpt, "test/base", _LORA_SHAPE) is False

    def test_donor_checkpoint_valid_false_when_base_id_none(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        assert donor_checkpoint_valid(ckpt, None, _LORA_SHAPE) is False

    def test_donor_checkpoint_valid_true_on_matching_base_and_shape(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        _write_donor_slot(ckpt, base_model_id="test/base", lora_shape=_LORA_SHAPE)
        assert donor_checkpoint_valid(ckpt, "test/base", _LORA_SHAPE) is True

    def test_donor_checkpoint_valid_false_on_base_mismatch(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        _write_donor_slot(ckpt, base_model_id="other/base", lora_shape=_LORA_SHAPE)
        assert donor_checkpoint_valid(ckpt, "test/base", _LORA_SHAPE) is False

    def test_donor_checkpoint_valid_false_on_lora_shape_mismatch(self, tmp_path):
        """M4: an operator rank edit invalidates a checkpoint trained at the
        old shape, forcing a rebuild instead of crashing later inside
        copy_adapter_weights_subset on a tensor-shape mismatch."""
        ckpt = donor_checkpoint_dir(tmp_path)
        _write_donor_slot(
            ckpt,
            base_model_id="test/base",
            lora_shape={"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]},
        )
        assert donor_checkpoint_valid(ckpt, "test/base", _LORA_SHAPE) is False

    def test_donor_checkpoint_valid_false_on_unparseable_meta(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        _write_donor_slot(ckpt, meta_text="{not valid json")
        assert donor_checkpoint_valid(ckpt, "test/base", _LORA_SHAPE) is False

    def test_donor_checkpoint_valid_false_on_missing_safetensors(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        _write_donor_slot(ckpt, base_model_id="test/base", write_weights=False)
        assert donor_checkpoint_valid(ckpt, "test/base", _LORA_SHAPE) is False

    def test_latest_donor_slot_picks_newest_and_skips_dot_dirs(self, tmp_path):
        ckpt = donor_checkpoint_dir(tmp_path)
        (ckpt / "20260101-000000").mkdir(parents=True)
        (ckpt / "20260201-000000").mkdir(parents=True)
        (ckpt / ".training_scratch").mkdir(parents=True)
        (ckpt / ".pending").mkdir(parents=True)
        assert _latest_donor_slot(ckpt).name == "20260201-000000"

    def test_latest_donor_slot_none_when_dir_absent(self, tmp_path):
        assert _latest_donor_slot(donor_checkpoint_dir(tmp_path)) is None


def _fake_atomic_save_adapter(model, target_dir, adapter_name):
    """Stand-in for paramem.models.loader.atomic_save_adapter: writes a
    real (tiny, fake-content) slot so the donor's SHA-256 + meta write can
    run against actual bytes on disk, without touching PEFT/torch save."""
    target_dir = Path(target_dir)
    slot = target_dir / "20260726-000000"
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "adapter_model.safetensors").write_bytes(b"fake-donor-weights")
    return slot


def _fake_create_adapter(model, cfg, name):
    """Stand-in for paramem.models.loader.create_adapter that ACTUALLY
    registers *name* in peft_config (M1 fix) -- the prior fake
    (``lambda m, cfg, name: m``) never added the transient name, which made
    every "transient slot deleted/not-deleted" assertion in this test class
    vacuously true regardless of whether the finally-cleanup ran at all."""
    model.peft_config[name] = MagicMock()
    return model


class TestBuildDonor:
    """build_donor orchestrates: sweep-orphan-pending -> transient create ->
    _train_tier_adapter (the shared funnel) -> abort/empty check ->
    atomic_save_adapter -> meta write -> prune -> transient delete, the
    last step ALWAYS in a finally regardless of outcome."""

    def test_trains_through_shared_funnel_with_no_special_budget(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop._train_tier_adapter = MagicMock(return_value=({"aborted": False}, None))

        with (
            patch(
                "paramem.models.loader.atomic_save_adapter",
                side_effect=_fake_atomic_save_adapter,
            ),
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.switch_adapter"),
        ):
            build_donor(loop, seed=1, n=DONOR_MIN_ENTRIES)

        assert loop._train_tier_adapter.called
        call_kwargs = loop._train_tier_adapter.call_args.kwargs
        assert call_kwargs["adapter_name"] == DONOR_BUILD_ADAPTER_NAME
        assert call_kwargs["adapter_config"] is loop.episodic_config
        # Same funnel, same config object -- no special-case budget threaded in.
        assert call_kwargs["training_config"] is loop.training_config
        # Transient slot MEANINGFULLY deleted (the fake create_adapter above
        # actually registers it, so this assertion is not vacuous).
        assert DONOR_BUILD_ADAPTER_NAME not in loop.model.peft_config

    def test_persists_meta_with_seed_recipe_triples_sha_base_id_and_lora_shape(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop._train_tier_adapter = MagicMock(return_value=({"aborted": False}, None))

        with (
            patch(
                "paramem.models.loader.atomic_save_adapter",
                side_effect=_fake_atomic_save_adapter,
            ),
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.switch_adapter"),
        ):
            slot = build_donor(loop, seed=42, n=DONOR_MIN_ENTRIES)

        meta = json.loads((slot / "donor_meta.json").read_text())
        assert meta["seed"] == 42
        assert meta["base_model_id"] == "test/base-model"
        assert meta["recipe"]
        assert len(meta["triples"]) >= DONOR_MIN_ENTRIES
        assert meta["lora_shape"] == {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}
        expected_sha = hashlib.sha256((slot / "adapter_model.safetensors").read_bytes()).hexdigest()
        assert meta["weights_sha256"] == expected_sha

    def test_transient_slot_deleted_even_on_training_failure(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop._train_tier_adapter = MagicMock(side_effect=RuntimeError("boom"))

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.switch_adapter"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            build_donor(loop, seed=1, n=DONOR_MIN_ENTRIES)

        assert DONOR_BUILD_ADAPTER_NAME not in loop.model.peft_config

    def test_aborted_training_raises_and_writes_no_checkpoint(self, tmp_path):
        """H2: an aborted (thermal/pause) run must never persist a
        checkpoint -- that would seed every future measured-cold fold from
        a run that never actually trained, forever."""
        loop = _make_bare_loop(tmp_path)
        loop._train_tier_adapter = MagicMock(return_value=({"aborted": True}, None))

        with (
            patch("paramem.models.loader.atomic_save_adapter") as mock_save,
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.switch_adapter"),
            pytest.raises(DonorBuildIncomplete),
        ):
            build_donor(loop, seed=1, n=DONOR_MIN_ENTRIES)

        assert not mock_save.called
        assert not donor_checkpoint_dir(tmp_path).exists() or not any(
            donor_checkpoint_dir(tmp_path).iterdir()
        )
        assert DONOR_BUILD_ADAPTER_NAME not in loop.model.peft_config

    def test_none_metrics_raises_and_writes_no_checkpoint(self, tmp_path):
        """H2: (None, None) (no training examples) must also be treated as
        incomplete, never persisted."""
        loop = _make_bare_loop(tmp_path)
        loop._train_tier_adapter = MagicMock(return_value=(None, None))

        with (
            patch("paramem.models.loader.atomic_save_adapter") as mock_save,
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.switch_adapter"),
            pytest.raises(DonorBuildIncomplete),
        ):
            build_donor(loop, seed=1, n=DONOR_MIN_ENTRIES)

        assert not mock_save.called
        assert DONOR_BUILD_ADAPTER_NAME not in loop.model.peft_config

    def test_prunes_older_slots_after_successful_save(self, tmp_path):
        """L6: exactly one donor artifact persists at a time."""
        loop = _make_bare_loop(tmp_path)
        loop._train_tier_adapter = MagicMock(return_value=({"aborted": False}, None))
        checkpoint_root = donor_checkpoint_dir(tmp_path)
        stale_slot = checkpoint_root / "20200101-000000"
        stale_slot.mkdir(parents=True)
        (stale_slot / "adapter_model.safetensors").write_bytes(b"stale")

        with (
            patch(
                "paramem.models.loader.atomic_save_adapter",
                side_effect=_fake_atomic_save_adapter,
            ),
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.switch_adapter"),
        ):
            slot = build_donor(loop, seed=1, n=DONOR_MIN_ENTRIES)

        remaining = {p.name for p in checkpoint_root.iterdir() if p.is_dir()}
        assert remaining == {slot.name}


class TestSeedingHook:
    """ConsolidationLoop._maybe_seed_from_donor / the funnel's gating logic."""

    def test_flag_off_never_calls_seeding(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=False)
        loop._maybe_seed_from_donor = MagicMock()
        loop._indexed_dataset = MagicMock(return_value=MagicMock())
        loop._enable_gradient_checkpointing = MagicMock()
        loop._maybe_make_recall_callback = MagicMock(return_value=(None, None))
        loop._build_training_hooks = MagicMock(return_value=MagicMock())

        with (
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.1, "aborted": False},
            ),
        ):
            metrics, _ = loop._train_tier_adapter(
                [{"key": "graph1", "subject": "s", "predicate": "p", "object": "o"}],
                adapter_name="episodic",
                adapter_config=loop.episodic_config,
                training_config=loop.training_config,
                output_dir=tmp_path / "scratch",
                run_name="test",
                phase_name="test",
            )

        assert not loop._maybe_seed_from_donor.called
        assert metrics["donor_seeded"] is False

    def test_cold_valid_checkpoint_loads_copies_and_deletes_transient_in_order(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        # measured_adapter_init_state calls param.data.norm().item() -- a real
        # zero tensor gives a faithful "cold" read (no MagicMock chain).
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]
        loop.model.active_adapter = DONOR_LOAD_ADAPTER_NAME

        call_order: list[str] = []

        def _load_side_effect(model, checkpoint_dir, transient_name):
            # Mirrors load_donor_into_transient_slot's real side effect (mounts
            # the transient adapter) so the finally-block delete path below
            # has something to find and remove.
            call_order.append("load")
            model.peft_config[transient_name] = MagicMock()

        def _copy_side_effect(model, src, dst):
            call_order.append("copy")
            return 4

        def _delete_side_effect(name):
            call_order.append("delete")
            del loop.model.peft_config[name]

        loop.model.delete_adapter.side_effect = _delete_side_effect

        with (
            patch("paramem.training.donor.donor_checkpoint_valid", return_value=True),
            patch(
                "paramem.training.donor.load_donor_into_transient_slot",
                side_effect=_load_side_effect,
            ) as mock_load,
            patch(
                "paramem.models.loader.copy_adapter_weights_subset",
                side_effect=_copy_side_effect,
            ) as mock_copy,
        ):
            seeded = loop._maybe_seed_from_donor("episodic")

        assert seeded is True
        assert call_order == ["load", "copy", "delete"]
        assert mock_load.called
        assert mock_copy.called
        assert DONOR_LOAD_ADAPTER_NAME not in loop.model.peft_config

    def test_transient_deleted_when_copy_raises(self, tmp_path):
        """M1: a copy_adapter_weights_subset failure (e.g. a genuine
        tensor-shape mismatch the lora_shape check missed) must still leave
        the transient slot cleaned up, not leaked."""
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]
        loop.model.active_adapter = DONOR_LOAD_ADAPTER_NAME

        def _load_side_effect(model, checkpoint_dir, transient_name):
            model.peft_config[transient_name] = MagicMock()

        def _delete_side_effect(name):
            del loop.model.peft_config[name]

        loop.model.delete_adapter.side_effect = _delete_side_effect

        with (
            patch("paramem.training.donor.donor_checkpoint_valid", return_value=True),
            patch(
                "paramem.training.donor.load_donor_into_transient_slot",
                side_effect=_load_side_effect,
            ),
            patch(
                "paramem.models.loader.copy_adapter_weights_subset",
                side_effect=RuntimeError("shape mismatch"),
            ),
            pytest.raises(RuntimeError, match="shape mismatch"),
        ):
            loop._maybe_seed_from_donor("episodic")

        assert DONOR_LOAD_ADAPTER_NAME not in loop.model.peft_config

    def test_base_id_mismatch_skips_without_crash(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with (
            patch("paramem.training.donor.donor_checkpoint_valid", return_value=False),
            patch("paramem.training.donor.build_donor") as mock_build,
            patch("paramem.training.donor.load_donor_into_transient_slot") as mock_load,
            patch("paramem.models.loader.copy_adapter_weights_subset") as mock_copy,
        ):
            seeded = loop._maybe_seed_from_donor("episodic")

        assert seeded is False
        assert mock_build.called, "a mismatched/missing checkpoint must attempt a rebuild"
        assert not mock_load.called
        assert not mock_copy.called

    def test_donor_build_incomplete_skips_without_crash(self, tmp_path):
        """H2: when build_donor cannot complete this fold (thermal/pause
        abort inside the donor's own training), the seeding hook must skip
        cleanly -- not propagate the exception into the caller's fold."""
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with (
            patch("paramem.training.donor.donor_checkpoint_valid", return_value=False),
            patch(
                "paramem.training.donor.build_donor",
                side_effect=DonorBuildIncomplete("aborted"),
            ) as mock_build,
            patch("paramem.training.donor.load_donor_into_transient_slot") as mock_load,
        ):
            seeded = loop._maybe_seed_from_donor("episodic")

        assert seeded is False
        assert mock_build.called
        assert not mock_load.called

    def test_unresolvable_base_id_skips_without_crash(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        loop.model.get_base_model.return_value.config._name_or_path = None
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with patch("paramem.training.donor.build_donor") as mock_build:
            seeded = loop._maybe_seed_from_donor("episodic")

        assert seeded is False
        assert not mock_build.called

    def test_warm_target_never_seeds(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.ones(2, 2)),
        ]

        with patch("paramem.training.donor.donor_checkpoint_valid") as mock_valid:
            seeded = loop._maybe_seed_from_donor("episodic")

        assert seeded is False
        assert not mock_valid.called, "a warm target must never even check the checkpoint"

    def test_build_slot_name_excluded_from_seeding_hook(self, tmp_path):
        """Training the donor's own transient build slot must never
        recursively re-trigger donor seeding on itself."""
        loop = _make_bare_loop(tmp_path)
        loop._indexed_dataset = MagicMock(return_value=MagicMock())
        loop._enable_gradient_checkpointing = MagicMock()
        loop._maybe_make_recall_callback = MagicMock(return_value=(None, None))
        loop._build_training_hooks = MagicMock(return_value=MagicMock())
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        loop._maybe_seed_from_donor = MagicMock()

        with (
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.1, "aborted": False},
            ),
        ):
            loop._train_tier_adapter(
                [{"key": "graph1", "subject": "s", "predicate": "p", "object": "o"}],
                adapter_name=DONOR_BUILD_ADAPTER_NAME,
                adapter_config=loop.episodic_config,
                training_config=loop.training_config,
                output_dir=tmp_path / "scratch",
                run_name="test",
                phase_name="test",
            )

        assert not loop._maybe_seed_from_donor.called

    def test_funnel_tags_metrics_donor_seeded_true_when_seeded(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop._indexed_dataset = MagicMock(return_value=MagicMock())
        loop._enable_gradient_checkpointing = MagicMock()
        loop._maybe_make_recall_callback = MagicMock(return_value=(None, None))
        loop._build_training_hooks = MagicMock(return_value=MagicMock())
        loop.training_config = TrainingConfig(donor_seeding_enabled=True)
        loop._maybe_seed_from_donor = MagicMock(return_value=True)

        with (
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.1, "aborted": False},
            ),
        ):
            metrics, _ = loop._train_tier_adapter(
                [{"key": "graph1", "subject": "s", "predicate": "p", "object": "o"}],
                adapter_name="episodic",
                adapter_config=loop.episodic_config,
                training_config=loop.training_config,
                output_dir=tmp_path / "scratch",
                run_name="test",
                phase_name="test",
            )

        assert metrics["donor_seeded"] is True
        loop._maybe_seed_from_donor.assert_called_once_with("episodic")


class TestLoadDonorIntoTransientSlot:
    def test_raises_when_no_checkpoint(self, tmp_path):
        model = MagicMock()
        with pytest.raises(FileNotFoundError):
            load_donor_into_transient_slot(model, donor_checkpoint_dir(tmp_path), "_x")
