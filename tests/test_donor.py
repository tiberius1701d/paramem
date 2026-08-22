"""Tests for the donor-adapter lifecycle (paramem.training.donor).

CPU-only. No GPU, no real model weights -- ``build_donor``/the seeding
hook are exercised against mocked PEFT primitives and a bare
``ConsolidationLoop`` built via ``object.__new__`` (same pattern
``tests/test_consolidation.py`` already uses for funnel-level unit tests),
never a real ``from_pretrained`` load.

Covers donor entries/topology-id/triples-hash, ``resolve_donor_checkpoint``,
loading a donor into a transient slot, the borrowed-donor cache, and
``donor_checkpoint_valid`` across key-lifecycle events (``TestDonorFixture``
through ``TestDonorCheckpointValidityAcrossKeyLifecycleEvents`` below).
``donor_checkpoint_valid``/``build_donor`` validation against a real
manifest-shaped slot is covered in
``tests/test_test20_donor_validation_arms.py``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.donor import (
    _FIXTURE_PATH,
    DONOR_BUILD_ADAPTER_NAME,
    DONOR_KEY_BAND_WIDTH,
    DONOR_META_FILENAME,
    DONOR_MIN_ENTRIES,
    DONOR_RECIPE_ID,
    DonorBuildIncomplete,
    donor_entries,
    donor_slot_valid,
    donor_store_dir,
    donor_topology_id,
    iter_donor_stores,
    load_donor_into_transient_slot,
    triples_hash,
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
_PROC_LORA_SHAPE = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "gate_proj"]}
_BASE_ID = "test/base-model"


def _make_bare_loop(tmp_path: Path) -> ConsolidationLoop:
    """Minimal ConsolidationLoop stub with only the attributes
    ``_train_tier_adapter`` / ``_resolve_donor_checkpoint`` read.

    Bypasses ``__init__`` (``object.__new__``) to avoid any model/GPU
    requirement -- the same pattern the per-class ``_make_loop`` helpers in
    ``tests/test_consolidation.py``, ``tests/test_fold_build_driver.py`` and
    ``tests/test_fold_phase1.py`` already use.

    Carries all three tier configs (episodic/semantic/procedural) --
    ``build_donor``'s dead-topology pruning derives its live-id set from
    all three, and episodic/semantic deliberately share one topology
    (the attention-only shape) while procedural is the second, so
    cross-topology tests need both shapes present.
    """
    from peft import PeftModel

    loop = object.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.model.__class__ = PeftModel
    loop.model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "procedural": MagicMock(),
    }
    # Real dict mutation on delete -- makes "transient slot deleted/not
    # deleted" assertions meaningful: the prior fake create_adapter never
    # registered the transient name, so with a MagicMock delete_adapter
    # that does nothing either, those assertions passed whether or not the
    # cleanup code ran at all.
    loop.model.delete_adapter.side_effect = lambda name: loop.model.peft_config.pop(name, None)
    # Real active_adapter mutation on switch -- drop_adapter_slot re-checks
    # active_adapter_name(model) after its switch attempt (treats a switch
    # that does not actually land the same as a failed one), so a bare
    # no-op set_adapter would make every switch-then-delete assertion here
    # vacuous (the delete would be skipped as "switch did not land" even
    # though nothing about the fake actually failed).
    loop.model.set_adapter.side_effect = lambda name: setattr(loop.model, "active_adapter", name)
    loop.model.get_base_model.return_value.config._name_or_path = _BASE_ID
    loop.tokenizer = MagicMock()
    loop.training_config = TrainingConfig()
    loop.tier_adapters = {
        "episodic": AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"]),
        "semantic": AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"]),
        "procedural": AdapterConfig(rank=8, alpha=16, target_modules=["q_proj", "gate_proj"]),
    }
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop._thermal_policy = None
    return loop


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


class TestTriplesHash:
    """triples_hash is a pure, order-independent canonical hash."""

    def test_order_independent_for_shuffled_input(self):
        entries = donor_entries(seed=3, n=DONOR_MIN_ENTRIES)
        shuffled = list(entries)
        random.Random(99).shuffle(shuffled)
        assert shuffled != entries, "shuffle must actually reorder for this to be a real test"
        assert triples_hash(entries) == triples_hash(shuffled)


class TestDonorTopologyId:
    """donor_topology_id: the canonical, filesystem-safe topology identity
    over {r, lora_alpha, target_modules} -- the SAME fields
    ensure_adapter_matching (paramem.models.loader) already treats as shape
    identity."""

    def test_topology_id_is_order_insensitive_on_target_modules(self):
        """Reordering target_modules must yield the SAME id -- module
        order is not a topology difference."""
        a = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]}
        b = {"r": 8, "lora_alpha": 16, "target_modules": ["o_proj", "k_proj", "v_proj", "q_proj"]}
        assert donor_topology_id(a) == donor_topology_id(b)

    def test_topology_id_separates_rank_alpha_and_module_set(self):
        attn = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
        rank16 = {**attn, "r": 16, "lora_alpha": 32}
        full = {
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
        ids = {donor_topology_id(attn), donor_topology_id(rank16), donor_topology_id(full)}
        assert len(ids) == 3, "rank/alpha edit and module-set edit must each produce a distinct id"

    def test_shipped_config_literals_match_the_documented_derivation(self):
        """Pins the two literals the plan's re-derivation step depends on --
        a mismatch here means the recipe (rank/alpha normalization, sort,
        digest length) diverged from what production directories are keyed
        by."""
        attn = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
        full = {
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
        assert donor_topology_id(attn) == "r8-a16-4mod-003cf56e"
        assert donor_topology_id(full) == "r8-a16-7mod-b102f771"

    def test_episodic_and_semantic_share_a_topology_procedural_does_not(self):
        """Pins "2 donors, not 3": built from the three real AdapterConfigs
        the shipped test fixture carries -- episodic and semantic (both
        attention-only) collapse to ONE topology id; procedural
        (attention+MLP) is a distinct second one."""
        from paramem.models.loader import lora_shape_fields
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        tier_adapters = cfg.tier_config_map()
        episodic_id = donor_topology_id(lora_shape_fields(tier_adapters["episodic"]))
        semantic_id = donor_topology_id(lora_shape_fields(tier_adapters["semantic"]))
        procedural_id = donor_topology_id(lora_shape_fields(tier_adapters["procedural"]))

        assert episodic_id == semantic_id
        assert procedural_id != episodic_id

    def test_store_dir_is_topology_scoped_at_the_adapter_root(self, tmp_path):
        attn_dir = donor_store_dir(tmp_path, _BASE_ID, _LORA_SHAPE)
        proc_dir = donor_store_dir(tmp_path, _BASE_ID, _PROC_LORA_SHAPE)

        assert attn_dir.parent == tmp_path
        assert donor_topology_id(_LORA_SHAPE) in attn_dir.name
        assert attn_dir != proc_dir


class TestResolveDonorCheckpoint:
    """ConsolidationLoop._resolve_donor_checkpoint / the funnel's donor
    resolution and gating logic.

    Donor resolution is the unconditional standard mechanism (no feature
    flag -- the prior donor_seeding_enabled flag was retired once the
    validation arms passed; see benchmarking.md). This method resolves and
    validates a checkpoint directory -- it copies NOTHING; the copy into the
    transient staging slot, and the load-failure degrade, are
    ``train_adapter``'s own job (see ``tests/test_staging_adapter.py``'s
    staging-init pins). Every test below builds a plain ``TrainingConfig()``.
    """

    def test_resolves_valid_checkpoint_for_a_measured_cold_target(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig()
        # has_prior_trained_weights measures via param.data.norm().item() --
        # a real zero tensor gives a faithful "cold" (no prior weights) read.
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with patch("paramem.training.donor.donor_checkpoint_valid", return_value=True):
            resolved = loop._resolve_donor_checkpoint("episodic", loop.tier_adapters["episodic"])

        expected_dir = donor_store_dir(tmp_path, _BASE_ID, _LORA_SHAPE)
        assert resolved == expected_dir

    def test_resolves_the_targets_own_topology_dir_for_procedural(self, tmp_path):
        """Resolving a procedural (7-module) target resolves the PROCEDURAL
        topology directory, never episodic/semantic's."""
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig()
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.procedural.weight", torch.zeros(2, 2)),
        ]

        with patch("paramem.training.donor.donor_checkpoint_valid", return_value=True):
            resolved = loop._resolve_donor_checkpoint(
                "procedural", loop.tier_adapters["procedural"]
            )

        expected_dir = donor_store_dir(tmp_path, _BASE_ID, _PROC_LORA_SHAPE)
        assert resolved == expected_dir

    def test_base_id_mismatch_skips_without_crash(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig()
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with (
            patch("paramem.training.donor.donor_checkpoint_valid", return_value=False),
            patch("paramem.training.donor.build_donor") as mock_build,
        ):
            resolved = loop._resolve_donor_checkpoint("episodic", loop.tier_adapters["episodic"])

        assert resolved is None
        assert mock_build.called, "a mismatched/missing checkpoint must attempt a rebuild"

    def test_donor_build_incomplete_skips_without_crash(self, tmp_path):
        """H2: when build_donor cannot complete this fold (thermal/pause
        abort inside the donor's own training), resolution must degrade
        cleanly -- not propagate the exception into the caller's fold."""
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig()
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
        ):
            resolved = loop._resolve_donor_checkpoint("episodic", loop.tier_adapters["episodic"])

        assert resolved is None
        assert mock_build.called

    def test_unresolvable_base_id_skips_without_crash(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig()
        loop.model.get_base_model.return_value.config._name_or_path = None
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with patch("paramem.training.donor.build_donor") as mock_build:
            resolved = loop._resolve_donor_checkpoint("episodic", loop.tier_adapters["episodic"])

        assert resolved is None
        assert not mock_build.called

    def test_prior_trained_weights_resolve_no_donor(self, tmp_path):
        """A tier that ``has_prior_trained_weights`` (measures warm) never
        resolves a donor -- warm-start always wins over seeding."""
        loop = _make_bare_loop(tmp_path)
        loop.training_config = TrainingConfig()
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.ones(2, 2)),
        ]

        with patch("paramem.training.donor.donor_checkpoint_valid") as mock_valid:
            resolved = loop._resolve_donor_checkpoint("episodic", loop.tier_adapters["episodic"])

        assert resolved is None
        assert not mock_valid.called, "a warm target must never even check the checkpoint"

    def test_build_slot_name_excluded_from_donor_resolution(self, tmp_path):
        """Training the donor's own transient build slot must never
        recursively re-trigger donor resolution on itself."""
        loop = _make_bare_loop(tmp_path)
        loop._indexed_dataset = MagicMock(return_value=MagicMock())
        loop._enable_gradient_checkpointing = MagicMock()
        loop._maybe_make_recall_callback = MagicMock(return_value=(None, None))
        loop._build_training_hooks = MagicMock(return_value=MagicMock())
        loop.training_config = TrainingConfig()
        loop._resolve_donor_checkpoint = MagicMock()

        with (
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.1, "aborted": False, "init": "cold"},
            ),
        ):
            loop._train_tier_adapter(
                [{"key": "graph1", "subject": "s", "predicate": "p", "object": "o"}],
                adapter_name=DONOR_BUILD_ADAPTER_NAME,
                adapter_config=loop.tier_adapters["episodic"],
                training_config=loop.training_config,
                output_dir=tmp_path / "scratch",
                run_name="test",
                phase_name="test",
            )

        assert not loop._resolve_donor_checkpoint.called

    def test_funnel_threads_the_resolved_checkpoint_and_carries_init_donor(self, tmp_path):
        """The funnel passes _resolve_donor_checkpoint's return value to
        train_adapter as donor_checkpoint_dir, and returns whatever
        train_adapter reports as metrics["init"] unchanged -- the fact that
        seeding actually applied is reported once, by train_adapter, never
        re-derived here."""
        loop = _make_bare_loop(tmp_path)
        loop._indexed_dataset = MagicMock(return_value=MagicMock())
        loop._enable_gradient_checkpointing = MagicMock()
        loop._maybe_make_recall_callback = MagicMock(return_value=(None, None))
        loop._build_training_hooks = MagicMock(return_value=MagicMock())
        loop.training_config = TrainingConfig()
        resolved_dir = tmp_path / "donor-store"
        loop._resolve_donor_checkpoint = MagicMock(return_value=resolved_dir)

        captured_kwargs: dict = {}

        def _fake_train_adapter(**kwargs):
            captured_kwargs.update(kwargs)
            return {"train_loss": 0.1, "aborted": False, "init": "donor"}

        with (
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                side_effect=_fake_train_adapter,
            ),
        ):
            metrics, _ = loop._train_tier_adapter(
                [{"key": "graph1", "subject": "s", "predicate": "p", "object": "o"}],
                adapter_name="episodic",
                adapter_config=loop.tier_adapters["episodic"],
                training_config=loop.training_config,
                output_dir=tmp_path / "scratch",
                run_name="test",
                phase_name="test",
            )

        assert metrics["init"] == "donor"
        loop._resolve_donor_checkpoint.assert_called_once_with(
            "episodic", loop.tier_adapters["episodic"]
        )
        assert captured_kwargs["donor_checkpoint_dir"] == resolved_dir


class TestLoadDonorIntoTransientSlot:
    def test_raises_when_no_checkpoint(self, tmp_path):
        model = MagicMock()
        with pytest.raises(FileNotFoundError):
            load_donor_into_transient_slot(
                model, donor_store_dir(tmp_path, _BASE_ID, _LORA_SHAPE), "_x"
            )


class TestBorrowedDonorCache:
    """A loop whose output_dir is a scratch tree (a migration trial) reads the
    live deployment's donor stores instead of rebuilding them, and can never
    write to what it borrows."""

    def test_donor_root_defaults_to_the_loops_own_output_dir(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        assert loop.donor_adapter_root == loop.output_dir

    def test_borrowing_redirects_resolution_to_the_lent_root(self, tmp_path):
        live_root = tmp_path / "live"
        live_root.mkdir()
        loop = _make_bare_loop(tmp_path / "trial")
        loop.borrow_donor_cache(live_root)
        assert loop.donor_adapter_root == live_root

    def test_borrowed_loop_never_builds_into_a_cache_it_does_not_own(self, tmp_path):
        """The base-swap trial case: nothing valid in the lent cache for this
        base/topology, so the resolver returns None (the target trains cold)
        rather than writing there."""
        live_root = tmp_path / "live"
        live_root.mkdir()
        loop = _make_bare_loop(tmp_path / "trial")
        loop.borrow_donor_cache(live_root)
        import torch

        loop.model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(2, 2)),
        ]

        with patch("paramem.training.donor.build_donor") as mock_build:
            resolved = loop._resolve_donor_checkpoint("episodic", loop.tier_adapters["episodic"])

        assert resolved is None
        mock_build.assert_not_called()
        assert iter_donor_stores(live_root) == []


class TestDonorCheckpointValidityAcrossKeyLifecycleEvents:
    """A donor checkpoint's validity must survive events that touch its
    encryption or its manifest schema without touching its content:
    ``donor_slot_valid`` verifies the WEIGHTS against the manifest's own
    plaintext payload digest (``manifest.payload.sha256``,
    :func:`~paramem.backup.hashing.plaintext_sha256`), which a daily-key
    rotation does not change, and the schema-v5 migration script rewrites
    metadata only -- the recorded digest tracks the same unchanged bytes on
    disk either way.

    Slots here are built directly on disk (weights + meta.json +
    donor_meta.json) -- no GPU, no PEFT/model load, matching
    ``donor_slot_valid``'s own filesystem/logic contract.
    """

    def test_donor_checkpoint_stays_valid_across_a_daily_key_rotation(self, tmp_path, monkeypatch):
        from pyrage import x25519

        from paramem.adapters.manifest import write_manifest
        from paramem.backup.encryption import envelope_encrypt_bytes, read_maybe_encrypted
        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )
        from tests._manifest_fixtures import BASE_MODEL_ID, LORA_SHAPE_DICT, make_train_manifest

        daily_a = mint_daily_identity()
        recovery = x25519.Identity.generate()
        daily_path = tmp_path / "daily_key.age"
        recovery_path = tmp_path / "recovery.pub"
        write_daily_key_file(wrap_daily_identity(daily_a, "pw"), daily_path)
        write_recovery_pub_file(recovery.to_public(), recovery_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)

        lora_shape = LORA_SHAPE_DICT
        store_dir = donor_store_dir(tmp_path, BASE_MODEL_ID, lora_shape)
        slot = store_dir / "20260101-000000"
        slot.mkdir(parents=True)

        seed, n = 1, DONOR_MIN_ENTRIES
        entries = donor_entries(seed, n)

        plaintext_weights = b"fake donor weights payload -- unchanged by rotation"
        digest = hashlib.sha256(plaintext_weights).hexdigest()
        (slot / "adapter_model.safetensors").write_bytes(envelope_encrypt_bytes(plaintext_weights))

        manifest = make_train_manifest(
            name=DONOR_BUILD_ADAPTER_NAME,
            registry_sha256="",
            key_count=len(entries),
            payload_sha256=digest,
        )
        write_manifest(slot, manifest)
        (slot / DONOR_META_FILENAME).write_text(
            json.dumps(
                {
                    "seed": seed,
                    "recipe": DONOR_RECIPE_ID,
                    "n_requested": n,
                    "triples": entries,
                    "triples_hash": triples_hash(entries),
                }
            )
        )

        assert donor_slot_valid(slot, BASE_MODEL_ID, lora_shape) is True

        # Rotate: decrypt with the OLD daily, re-encrypt the SAME plaintext
        # under a fresh daily identity -- the same content, a new key.
        plaintext_read_back = read_maybe_encrypted(slot / "adapter_model.safetensors")
        assert plaintext_read_back == plaintext_weights
        daily_b = mint_daily_identity()
        write_daily_key_file(wrap_daily_identity(daily_b, "pw"), daily_path)
        _clear_daily_identity_cache()
        weights_path = slot / "adapter_model.safetensors"
        weights_path.write_bytes(envelope_encrypt_bytes(plaintext_read_back))

        assert donor_slot_valid(slot, BASE_MODEL_ID, lora_shape) is True, (
            "a plaintext-content payload digest must survive re-encryption under a new daily key"
        )

    def test_donor_checkpoint_stays_valid_across_the_migration(self, tmp_path):
        # scripts/ is importable via pytest's pythonpath=["."] config; no
        # sys.path manipulation needed when only importing (not running
        # __main__).
        from scripts.migrate.stamp_slot_manifests_v5 import migrate
        from tests._manifest_fixtures import (
            BASE_MODEL_ID,
            LORA_SHAPE_DICT,
            v4_train_meta_dict,
            write_raw_meta,
        )

        adapter_root = tmp_path / "adapters"
        lora_shape = LORA_SHAPE_DICT
        store_dir = donor_store_dir(adapter_root, BASE_MODEL_ID, lora_shape)
        slot = store_dir / "20260101-000000"
        slot.mkdir(parents=True)

        seed, n = 1, DONOR_MIN_ENTRIES
        entries = donor_entries(seed, n)
        plaintext_weights = b"fake donor weights payload -- pre-migration"
        (slot / "adapter_model.safetensors").write_bytes(plaintext_weights)
        (slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')

        raw_meta = v4_train_meta_dict(
            name=DONOR_BUILD_ADAPTER_NAME, registry_sha256="", key_count=len(entries)
        )
        write_raw_meta(slot, raw_meta)
        (slot / DONOR_META_FILENAME).write_text(
            json.dumps(
                {
                    "seed": seed,
                    "recipe": DONOR_RECIPE_ID,
                    "n_requested": n,
                    "triples": entries,
                    "triples_hash": triples_hash(entries),
                }
            )
        )

        # Pre-migration: read_manifest refuses the prior (schema_version=4)
        # shape, so donor_slot_valid's never-raises contract degrades to
        # False rather than crashing the caller.
        assert donor_slot_valid(slot, BASE_MODEL_ID, lora_shape) is False

        migrate(adapter_root, dry_run=False)

        assert donor_slot_valid(slot, BASE_MODEL_ID, lora_shape) is True, (
            "the migrated donor manifest must validate without a rebuild"
        )
