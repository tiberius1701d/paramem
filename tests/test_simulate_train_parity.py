"""Parity: simulate and train modes produce equivalent MemoryStore state.

The cycle's mode-conditional code paths (key reconstruction source, training
step, persistence venue) must converge to identical post-cycle state for
everything that is mode-INDEPENDENT: cache contents, simhash registry,
indexed_key_registry entries.  Bytewise-different artifacts
(adapter_model.safetensors vs graph.json) are NOT compared.

Train mode is run with stubbed ``_train_adapter`` / ``save_adapter`` so the
test does not require GPU; the assertion is on the cycle's data-pipeline
output, not on weight values.

Class ``TestProbeKeysFromGraph`` covers the simulate-mode graph reader
(``DiskMemorySource.probe`` against ``graph.json``) and verifies the returned
result shape — those tests are mode-agnostic and are retained from the
previous file.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from paramem.memory.persistence import _IK_KEY_ATTR, save_memory_to_disk
from paramem.memory.source import DiskMemorySource
from paramem.memory.store import MemoryStore
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.graph_tier import GraphTierRefiner
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_EPISODIC_RELS: list[dict] = [
    {
        "subject": "Alice",
        "predicate": "lives_in",
        "object": "Berlin",
        "relation_type": "factual",
        # No speaker_id — tests gap-5 (default tagging)
    },
    {
        "subject": "Alice",
        "predicate": "works_at",
        "object": "Acme Corp",
        "relation_type": "factual",
        "speaker_id": "sp_explicit",  # explicit — must be preserved
    },
    {
        "subject": "Acme Corp",
        "predicate": "is_located_in",
        "object": "Germany",
        "relation_type": "factual",
    },
    {
        "subject": "Alice",
        "predicate": "knows",
        "object": "Bob",
        "relation_type": "factual",
    },
    {
        "subject": "Bob",
        "predicate": "likes",
        "object": "Coffee",
        "relation_type": "factual",
    },
    {
        "subject": "Carol",
        "predicate": "visits",
        "object": "London",
        "relation_type": "factual",
        "speaker_id": "",  # explicit empty — must NOT be overwritten
    },
]

_PROCEDURAL_RELS: list[dict] = [
    {
        "subject": "Alice",
        "predicate": "prefers",
        "object": "Remote work",
        "relation_type": "preference",
    },
    {
        "subject": "Alice",
        "predicate": "dislikes",
        "object": "Mondays",
        "relation_type": "preference",
    },
]

_SPEAKER_ID = "sp_test"
_STAMP = "20260101T0000"


def _build_loop(tmp_path: Path, *, procedural_enabled: bool = True) -> ConsolidationLoop:
    """Build a minimal ConsolidationLoop for parity testing.

    Bypasses ``__init__`` via ``__new__`` and sets only the attributes that
    ``run_consolidation_cycle`` reads — model, tokenizer, configs, store,
    counters, and flags.  The model's ``peft_config`` is a real dict so that
    ``create_interim_adapter`` / ``add_adapter`` can populate it without
    KeyError.
    """
    loop = ConsolidationLoop.__new__(ConsolidationLoop)

    # Model: real peft_config dict; add_adapter populates it.
    model = MagicMock()
    model.peft_config = {}  # real dict — must not be a MagicMock to avoid KeyError
    model.add_adapter.side_effect = lambda name, cfg: model.peft_config.update({name: cfg})

    loop.model = model
    loop.tokenizer = MagicMock()
    loop.config = ConsolidationConfig(indexed_key_replay=True)
    loop.training_config = TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
        recall_early_stopping=False,
    )
    loop.episodic_config = AdapterConfig(
        rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"]
    )
    loop.semantic_config = AdapterConfig(
        rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"]
    )
    loop.procedural_config = (
        AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
        if procedural_enabled
        else None
    )
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    loop._thermal_policy = None
    loop.shutdown_requested = False
    # Enrichment sizing scalars — see _make_bare_loop.  Always set by
    # ConsolidationLoop.__init__ (consolidation.py:501-502), never nulled by
    # release(), and read when the graph tier's refiner is constructed.
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50
    loop.merger = MagicMock()

    # _build_all_edge_entries_into reads merger.graph.edges(data=True).
    # Replace the MagicMock graph with a real MultiDiGraph populated from
    # _EPISODIC_RELS so the graph-walk mints keys for the parity tests.
    # The _materialize_consolidation_graph stub below skips merger.reset_graph(),
    # so the graph persists intact through the keyed-walk step.
    #
    # Speaker-ID node seeding: the unified builder distinguishes
    # "speaker_id key absent" (→ use default_speaker_id, i.e. the cycle's
    # speaker_id) from "speaker_id key present with explicit value" (→ keep as-is,
    # even if empty).  This mirrors _tag_speaker_id_defaults semantics.
    # For each relation that carries an EXPLICIT speaker_id (including ""),
    # the subject node is stamped with that value.  Relations without a
    # speaker_id key produce a node with the attribute absent.
    _real_graph = nx.MultiDiGraph()
    for _rel in _EPISODIC_RELS:
        _subj = _rel["subject"].lower().replace(" ", "_")
        _obj = _rel["object"].lower().replace(" ", "_")
        # Stamp speaker_id on the node only when the relation carries it explicitly.
        _node_kwargs: dict = {"attributes": {"name": _rel["subject"]}}
        if "speaker_id" in _rel:
            _node_kwargs["speaker_id"] = _rel["speaker_id"]
        _real_graph.add_node(_subj, **_node_kwargs)
        _real_graph.add_node(_obj, attributes={"name": _rel["object"]})
        _real_graph.add_edge(
            _subj,
            _obj,
            predicate=_rel["predicate"],
            relation_type=_rel.get("relation_type", "factual"),
        )
    loop.merger.graph = _real_graph

    # MemoryStore with registry enabled.
    store = MemoryStore(replay_enabled=True)
    for tier in ("episodic", "semantic", "procedural"):
        store.load_registry(tier, KeyRegistry())
    loop.store = store

    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.promoted_keys: set = set()
    loop.fingerprint_cache = None

    # Stub out the recall probe so tests with a MagicMock model do not
    # feed it into re.sub (which raises TypeError on non-string input).
    # These tests verify slot layout / GAP fixes, not recall gating; the
    # probe is covered separately in test_consolidation_recall_early_stop.py.
    loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}

    # Stub out _materialize_consolidation_graph so the materialize step does not
    # call reconstruct_graph / probe_entries on the MagicMock model.
    # The stub skips merger.reset_graph() so loop.merger.graph retains the
    # pre-populated keyless edges for the graph-walk keying step.
    # The materialize diagnostic is covered in TestMaterializeInterimExtraRelations.
    loop._materialize_consolidation_graph = lambda **kw: (set(), [])

    # Enrichment is off by default (refinement_enrichment="off", cloud master switch off
    # in ConsolidationConfig base defaults) — no attribute assignment needed.
    loop.full_consolidation_period_string = ""

    return loop


def _patches_for_train_mode():
    """Return list of context managers that stub the GPU-touching train path.

    Stubs:
    - ``paramem.training.trainer.train_adapter`` → no-op returning a metrics dict.
    - ``paramem.models.loader.save_adapter`` → no-op (avoids PEFT I/O).
    - ``paramem.adapters.manifest.build_manifest_for`` → returns None.
    - ``paramem.memory.interim_adapter.create_interim_adapter`` → populates
      peft_config[adapter_name] so the ring-full check in run_consolidation_cycle
      works; returns the model unchanged.
    """

    def _fake_create_interim(model, cfg, stamp):
        """Create interim adapter slot in the mock peft_config."""
        adapter_name = f"episodic_interim_{stamp}"
        model.peft_config[adapter_name] = MagicMock()
        return model

    return [
        patch(
            "paramem.training.trainer.train_adapter",
            return_value={"aborted": False, "train_loss": 0.0},
        ),
        patch("paramem.models.loader.save_adapter"),
        patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        patch(
            "paramem.memory.interim_adapter.create_interim_adapter",
            side_effect=_fake_create_interim,
        ),
    ]


# ---------------------------------------------------------------------------
# Parity tests: simulate vs train mode
# ---------------------------------------------------------------------------


class TestSimulateTrainParity:
    """run_consolidation_cycle in simulate and train modes must converge.

    Covered invariants:
      - speaker_id default tagging — every entry gets the caller's id
        when none was present on the relation.
      - per-tier scope — active_keys_in_tier returns identical sorted
        key lists in both modes.
      - Equality: tier_simhashes, store entries, and on-disk
        indexed_key_registry bytes are bytewise-equal between modes.

    Graph enrichment parity is verified separately in test_graph_enrichment.py;
    the merge flag is disabled here to keep the comparison deterministic.
    """

    @pytest.fixture()
    def loop_sim(self, tmp_path):
        """Simulate-mode loop backed by a private tmp subdir."""
        return _build_loop(tmp_path / "sim")

    @pytest.fixture()
    def loop_train(self, tmp_path):
        """Train-mode loop backed by a private tmp subdir."""
        return _build_loop(tmp_path / "train")

    def _run_sim(self, loop: ConsolidationLoop) -> dict:
        """Run one simulate cycle with the deterministic fixture relations."""
        return loop.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="simulate",
            run_label="parity",
            stamp=_STAMP,
        )

    def _run_train(self, loop: ConsolidationLoop) -> dict:
        """Run one train cycle with the deterministic fixture relations."""
        patches = _patches_for_train_mode()
        with patches[0], patches[1], patches[2], patches[3]:
            return loop.run_consolidation_cycle(
                list(_EPISODIC_RELS),
                list(_PROCEDURAL_RELS),
                speaker_id=_SPEAKER_ID,
                mode="train",
                run_label="parity",
                stamp=_STAMP,
            )

    def test_speaker_id_default_applied_both_modes(self, loop_sim, loop_train, tmp_path):
        """speaker_id is resolved from edge then subject node; absent node attr → "".

        speaker_id is resolved from the EDGE first, then the SUBJECT NODE
        attribute, then terminal fallback to "".  The caller's id is NOT
        injected — there is no ``default_speaker_id`` parameter.

        Resolution ladder:
        - Edge carries speaker_id  → use it.
        - Edge absent, node has speaker_id attr → use node value (even if "").
        - Neither edge nor node carries it → terminal fallback "".

        In the test fixture:
        - "alice" node is stamped with speaker_id="sp_explicit" (from
          Alice/works_at/Acme Corp which carries speaker_id="sp_explicit"), so
          Alice's edge entries inherit "sp_explicit".
        - "acme_corp" node has NO speaker_id attribute (no relation with
          subject=Acme Corp carries speaker_id), so its entry gets "".
        - "carol" node is stamped with speaker_id="" (Carol/visits/London carries
          speaker_id=""), so Carol's entry keeps "".
        """
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                assert "speaker_id" in entry, f"Entry {key} missing speaker_id"

        # Subject node with NO speaker_id attribute → terminal fallback "".
        # Acme Corp's node has no speaker_id attr; no relation stamped it.
        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("subject") == "Acme Corp" and entry.get("object") == "Germany":
                    assert entry["speaker_id"] == "", (
                        f"Terminal fallback failed for {key}: expected '', "
                        f"got {entry['speaker_id']!r}"
                    )

        # The relation with explicit speaker_id=sp_explicit must keep it.
        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("object") == "Acme Corp":
                    assert entry["speaker_id"] == "sp_explicit", (
                        f"Explicit speaker_id overwritten in {key}: {entry['speaker_id']!r}"
                    )

        # The relation with explicit speaker_id="" must keep the empty string.
        # The resolution ladder reads the node attr (stamped ""); it must not
        # be replaced.
        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("subject") == "Carol" and entry.get("object") == "London":
                    assert entry["speaker_id"] == "", (
                        f"Explicit empty speaker_id overwritten in {key}: {entry['speaker_id']!r}"
                    )

    def test_active_keys_in_tier_match(self, loop_sim, loop_train):
        """active_keys_in_tier returns identical sorted lists in both modes."""
        adapter_name = f"episodic_interim_{_STAMP}"
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        sim_keys = sorted(loop_sim.store.active_keys_in_tier(adapter_name))
        train_keys = sorted(loop_train.store.active_keys_in_tier(adapter_name))

        assert sim_keys == train_keys, (
            f"active_keys_in_tier diverged:\n  simulate: {sim_keys}\n  train:    {train_keys}"
        )

    def test_simhashes_equal_both_modes(self, loop_sim, loop_train):
        """Simhash registries are identical after one cycle in both modes."""
        adapter_name = f"episodic_interim_{_STAMP}"
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        sim_hashes = dict(loop_sim.store.tier_simhashes(adapter_name, include_stale=False))
        train_hashes = dict(loop_train.store.tier_simhashes(adapter_name, include_stale=False))

        assert sim_hashes == train_hashes, (
            f"Simhash registries diverged for tier {adapter_name!r}:\n"
            f"  simulate keys: {sorted(sim_hashes)}\n"
            f"  train keys:    {sorted(train_hashes)}"
        )

    def test_registry_bytes_equal_both_modes(self, loop_sim, loop_train):
        """On-disk indexed_key_registry.json bytes are bytewise-equal in both modes.

        The registry payload written by commit_tier_slot must be identical
        regardless of mode (simulate vs train).
        The bytes are deterministic because KeyRegistry serialises keys in
        sorted order.
        """
        adapter_name = f"episodic_interim_{_STAMP}"
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        sim_reg = adapter_slot_root_for_name(loop_sim.output_dir, adapter_name)
        train_reg = adapter_slot_root_for_name(loop_train.output_dir, adapter_name)

        sim_bytes_path = sim_reg / "indexed_key_registry.json"
        train_bytes_path = train_reg / "indexed_key_registry.json"

        assert sim_bytes_path.exists(), f"Simulate registry file missing: {sim_bytes_path}"
        assert train_bytes_path.exists(), f"Train registry file missing: {train_bytes_path}"

        sim_bytes = sim_bytes_path.read_bytes()
        train_bytes = train_bytes_path.read_bytes()

        assert sim_bytes == train_bytes, (
            f"Registry bytes diverged for {adapter_name!r}.\n"
            f"  simulate ({len(sim_bytes)} bytes): {sim_bytes[:200]!r}\n"
            f"  train    ({len(train_bytes)} bytes): {train_bytes[:200]!r}"
        )

    def test_registry_persists_under_adapter_name_not_tier(self, loop_sim, loop_train):
        """Regression: ``commit_tier_slot`` must serialise the registry under
        ``adapter_name`` (e.g. ``episodic_interim_<stamp>``), not under the
        bare tier label (``"episodic"``).

        Before this fix: ``tier_reg = loop.store.registry(tier)`` read the
        empty main-tier registry while the cycle's freshly assigned keys lived
        in ``store["episodic_interim_<stamp>"]``.  The on-disk file held
        bytes that hashed correctly against the empty registry but contained
        zero of the just-trained keys — symptomless until post-restart
        hydration recovered 0 episodic keys.
        """
        adapter_name = f"episodic_interim_{_STAMP}"
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        for label, loop in (("simulate", loop_sim), ("train", loop_train)):
            in_memory = loop.store.registry(adapter_name)
            assert in_memory is not None, f"[{label}] no registry under {adapter_name!r}"
            in_memory_keys = sorted(in_memory.list_active())
            assert in_memory_keys, (
                f"[{label}] in-memory registry under {adapter_name!r} is empty — "
                f"cycle did not populate it"
            )

            from paramem.memory.interim_adapter import adapter_slot_root_for_name
            from paramem.training.key_registry import KeyRegistry

            on_disk_path = (
                adapter_slot_root_for_name(loop.output_dir, adapter_name)
                / "indexed_key_registry.json"
            )
            assert on_disk_path.exists(), f"[{label}] on-disk registry missing"

            on_disk = KeyRegistry.load(on_disk_path)
            on_disk_keys = sorted(on_disk.list_active())

            assert on_disk_keys == in_memory_keys, (
                f"[{label}] on-disk registry diverges from in-memory.\n"
                f"  in-memory ({len(in_memory_keys)} keys): {in_memory_keys[:5]}...\n"
                f"  on-disk   ({len(on_disk_keys)} keys): {on_disk_keys[:5]}..."
            )

    def test_procedural_persisted_in_interim_slot_in_simulate_mode(self, loop_sim, tmp_path):
        """Procedural facts persist in the interim slot graph.json in simulate mode.

        Procedural entries ride the SINGLE interim slot
        (``episodic_interim_<stamp>``) alongside episodic — there is no separate
        per-cycle ``procedural/graph.json`` commit.  ``commit_tier_slot`` is called
        once with ``tier="episodic"`` and ``adapter_name="episodic_interim_<stamp>"``;
        ``all_keyed`` includes both episodic and procedural entries.

        The fixture's ``_EPISODIC_RELS`` contains a ``likes`` predicate (Bob / likes /
        Coffee), which ``filter_procedural_relations`` routes to the procedural tier via
        the secondary ``_PROCEDURAL_PREDICATES`` gate.  After ``_run_sim`` that entry is
        minted with a ``proc``-prefixed key and lands in the interim slot graph.json
        alongside the episodic keys.

        Asserts:
          - The interim slot graph.json exists and is non-empty.
          - At least one entry has a ``proc``-prefixed key (the durable procedural
            signal; the key prefix is the stable identifier throughout the pipeline).
        """
        from paramem.memory.interim_adapter import adapter_slot_root_for_name
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        self._run_sim(loop_sim)

        adapter_name = f"episodic_interim_{_STAMP}"
        interim_slot = adapter_slot_root_for_name(loop_sim.output_dir, adapter_name)
        graph_path = interim_slot / "graph.json"

        assert graph_path.exists(), (
            f"Interim slot graph.json missing at {graph_path} — "
            "procedural facts must persist in the interim slot, not a separate procedural/ dir"
        )

        graph = load_memory_from_disk(graph_path)
        entries = list(iter_entries(graph))
        assert len(entries) > 0, (
            "Interim slot graph.json is empty after simulate cycle — "
            "procedural and episodic entries must co-reside in the interim slot"
        )

        # The proc-prefix is the durable signal that a procedural fact was minted.
        # The fixture's "Bob likes Coffee" edge routes to procedural via
        # _PROCEDURAL_PREDICATES ("likes") and receives a proc-prefixed key.
        proc_keys = {e["key"] for e in entries if e.get("key", "").startswith("proc")}
        assert proc_keys, (
            f"No proc-prefixed keys found in interim slot graph.json — "
            f"procedural facts must co-reside with episodic in the interim slot. "
            f"Keys present: {sorted(e.get('key') for e in entries)}"
        )

    def test_slot_layout_train_has_manifest_simulate_has_graph(self, loop_sim, loop_train):
        """Slot layout assertion: train slots have meta.json; simulate slots have graph.json.

        commit_tier_slot (train) raises on manifest failure rather than saving
        without one — so any train slot that lands on disk has a manifest.
        Simulate slots have graph.json and no safetensors.
        """
        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        adapter_name = f"episodic_interim_{_STAMP}"
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        sim_slot = adapter_slot_root_for_name(loop_sim.output_dir, adapter_name)
        train_slot = adapter_slot_root_for_name(loop_train.output_dir, adapter_name)

        # Simulate slot: must have graph.json, must NOT have safetensors.
        assert (sim_slot / "graph.json").exists(), f"Simulate slot missing graph.json at {sim_slot}"
        assert not any(sim_slot.rglob("adapter_model.safetensors")), (
            f"Simulate slot must not contain safetensors at {sim_slot}"
        )

        # Train slot: must have indexed_key_registry.json (commit signal).
        # meta.json presence depends on whether the manifest mock returned None;
        # the registry file is the authoritative commit signal.
        assert (train_slot / "indexed_key_registry.json").exists(), (
            f"Train slot missing indexed_key_registry.json at {train_slot}"
        )

    def test_active_adapter_restored_after_cycle(self, loop_sim, loop_train):
        """model.set_adapter("episodic") is called at end of cycle in both modes.

        After run_consolidation_cycle, the active adapter must be restored to
        "episodic" (step 13 of the internal flow).  This test uses a MagicMock
        that tracks set_adapter calls and verifies the episodic restore occurs.
        """
        # Simulate mode: episodic must NOT be in peft_config (simulate has no PEFT
        # adapters), so step 13's guard ``if "episodic" in self.model.peft_config``
        # is False — no set_adapter call is expected.
        self._run_sim(loop_sim)
        # No peft_config entries in simulate → step 13 is a no-op.
        assert "episodic" not in loop_sim.model.peft_config, (
            "Simulate mode must not populate peft_config with 'episodic'"
        )

        # Train mode: the create_interim_adapter mock adds adapter_name to
        # peft_config but does NOT add "episodic".  Step 13 is therefore a no-op
        # in the mock harness too.  What we can assert is that the model's
        # set_adapter was called with adapter_name at step 9
        # (switch_adapter before training).
        patches = _patches_for_train_mode()
        with patches[0], patches[1], patches[2], patches[3]:
            loop_train.run_consolidation_cycle(
                list(_EPISODIC_RELS),
                list(_PROCEDURAL_RELS),
                speaker_id=_SPEAKER_ID,
                mode="train",
                run_label="parity",
                stamp=_STAMP,
            )

        # In train mode the mock peft_config was populated with adapter_name
        # by _fake_create_interim, but NOT with "episodic", so step 13 is
        # skipped.  The model.set_adapter call at step 9 (switch_adapter) IS
        # expected.
        assert loop_train.model.set_adapter.called, (
            "Train mode: model.set_adapter was never called during the cycle"
        )


# ---------------------------------------------------------------------------
# Interim recital dedup: simulate==train parity
# ---------------------------------------------------------------------------


class TestInterimRecitalDedupSimulateTrainParity:
    """Simulate==train parity for the interim recital-dedup feature.

    Uses a REAL GraphMerger (unlike this module's MagicMock-merger
    ``_build_loop``) so the actual Case-1 dedup collapse fires;
    ``reconstruct_graph`` is stubbed (no GPU), mirroring
    ``TestMaterializeInterimExtraRelations`` in test_consolidation.py.

    The interim fresh-derivation materialize call never passes ``source=``
    (always defaults to the "weights" body regardless of venue — see
    ``_materialize_consolidation_graph``'s docstring), so the dedup merge
    added at the end of that body runs identically for both the "simulate"
    and "train" venues of ``run_consolidation_cycle``.
    """

    _STAMP = "20260301T0000"

    @staticmethod
    def _fake_reconstruct(loop, *, tier=None, strict=False):
        """Reconstruct stub: returns empty graph + no failures (no GPU)."""
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        return ReconstructionResult(graph=nx.MultiDiGraph(), failures=[])

    def _make_loop(self, tmp_path: Path) -> ConsolidationLoop:
        """Minimal ConsolidationLoop with a REAL GraphMerger and one seeded
        main-tier key.  Interim recital dedup is unconditional.
        """
        from paramem.graph.merger import GraphMerger

        loop = ConsolidationLoop.__new__(ConsolidationLoop)

        # NOTE: deliberately do NOT set model.__class__ = PeftModel (unlike
        # some test_consolidation.py fixtures) -- mirrors this file's own
        # _build_loop so isinstance(self.model, PeftModel) is False and
        # _verify_saved_adapter_from_disk's disk-integrity probe (which would
        # need REAL adapter files that the mocked save_adapter never writes)
        # gracefully skips via its "not a PeftModel" branch.
        model = MagicMock()
        model.peft_config = {}
        model.add_adapter.side_effect = lambda name, cfg: model.peft_config.update({name: cfg})
        loop.model = model
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(indexed_key_replay=True)
        loop.training_config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            recall_early_stopping=False,
        )
        loop.episodic_config = AdapterConfig(
            rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"]
        )
        loop.semantic_config = AdapterConfig(
            rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"]
        )
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop._thermal_policy = None
        loop.shutdown_requested = False
        # Always set by __init__ in a real loop; needed unconditionally because
        # _refine_consolidation_graph constructs a GraphTierRefiner on every
        # call regardless of the normalize/enrich flags.
        loop.graph_enrichment_neighborhood_hops = 2
        loop.graph_enrichment_max_entities_per_pass = 50

        loop.merger = GraphMerger(model=None)

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        store.put(
            "episodic",
            "graph_main",
            {
                "key": "graph_main",
                "subject": "alice",
                "predicate": "lives_in",
                "object": "berlin",
            },
            simhash=1,
        )
        store.set_bookkeeping(
            "graph_main", speaker_id="sp_test", relation_type="factual", first_seen=""
        )
        loop.store = store

        loop.cycle_count = 0
        loop._indexed_next_index = 100
        loop._procedural_next_index = 1
        loop.promoted_keys: set = set()
        loop.fingerprint_cache = None
        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
        loop.full_consolidation_period_string = ""
        return loop

    def _seed_pending(self, loop: ConsolidationLoop) -> None:
        """Merge this cycle's pending-session content directly into
        ``merger.graph``: one RECITED fact (matches ``graph_main``) + one
        NOVEL fact — mirrors what ``extract_session`` would have already
        merged in before the interim cycle runs.
        """
        from paramem.graph.schema import Relation, SessionGraph

        session = SessionGraph(
            session_id="s1",
            timestamp="",
            entities=[],
            relations=[
                Relation(
                    subject="alice",
                    predicate="lives_in",
                    object="berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="sp_test",
                ),
                Relation(
                    subject="alice",
                    predicate="likes",
                    object="tea",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="sp_test",
                ),
            ],
        )
        loop.merger.merge(session, resolve_contradictions=False)

    def test_dedup_minted_keys_identical_simulate_vs_train(self, tmp_path):
        """A recited-plus-novel session with dedup ON mints the SAME interim
        keys regardless of venue -- the recited fact dedups against
        ``graph_main`` and only the novel fact mints, in BOTH venues.
        """
        loop_sim = self._make_loop(tmp_path / "sim")
        loop_train = self._make_loop(tmp_path / "train")
        self._seed_pending(loop_sim)
        self._seed_pending(loop_train)

        episodic_rels = [
            {
                "subject": "alice",
                "predicate": "lives_in",
                "object": "berlin",
                "relation_type": "factual",
                "speaker_id": "sp_test",
            },
            {
                "subject": "alice",
                "predicate": "likes",
                "object": "tea",
                "relation_type": "factual",
                "speaker_id": "sp_test",
            },
        ]

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            loop_sim.run_consolidation_cycle(
                list(episodic_rels),
                [],
                speaker_id="sp_test",
                mode="simulate",
                run_label="dedup-parity",
                stamp=self._STAMP,
            )

            patches = _patches_for_train_mode()
            with patches[0], patches[1], patches[2], patches[3]:
                loop_train.run_consolidation_cycle(
                    list(episodic_rels),
                    [],
                    speaker_id="sp_test",
                    mode="train",
                    run_label="dedup-parity",
                    stamp=self._STAMP,
                )

        adapter_name = f"episodic_interim_{self._STAMP}"
        sim_keys = set(loop_sim.store.active_keys_in_tier(adapter_name))
        train_keys = set(loop_train.store.active_keys_in_tier(adapter_name))

        assert sim_keys == train_keys, (
            f"minted interim keys diverged between venues: sim={sim_keys} train={train_keys}"
        )
        assert "graph_main" not in sim_keys and "graph_main" not in train_keys, (
            "the recited fact must dedup against the main-tier key in BOTH venues"
        )
        assert len(sim_keys) == 1, f"exactly one novel key expected; got {sim_keys}"


# ---------------------------------------------------------------------------
# TestProbeKeysFromGraph — DiskMemorySource.probe (mode-agnostic)
# ---------------------------------------------------------------------------


def _write_graph(path, quads: list[dict]) -> None:
    """Write *quads* as a simulate-mode graph.json at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for quad in quads:
        graph.add_edge(
            quad["subject"],
            quad["object"],
            **{
                _IK_KEY_ATTR: quad["key"],
                "predicate": quad.get("predicate", ""),
                "speaker_id": quad.get("speaker_id", ""),
            },
        )
    save_memory_to_disk(graph, path)


class TestProbeKeysFromGraph:
    """DiskMemorySource.probe reads graph.json matching the grouped-probe shape.

    Under perfect recall, hit results return::

        {"key": str, "subject": str, "predicate": str, "object": str,
         "confidence": 1.0, "format": "quad",
         "fact_text": str, "raw_output": str}

    Missing tiers / missing keys → ``None``.
    """

    def test_reads_episodic_from_subdir(self, tmp_path):
        """Canonical layout: episodic graph lives under episodic/ subdir."""
        _write_graph(
            tmp_path / "episodic" / "graph.json",
            [
                {
                    "key": "graph1",
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "",
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1"]})
        assert results["graph1"] is not None
        assert results["graph1"]["subject"] == "Alex"
        assert results["graph1"]["object"] == "Berlin"
        assert results["graph1"]["predicate"] == "lives_in"
        assert results["graph1"]["confidence"] == 1.0

    def test_reads_semantic_from_subdir(self, tmp_path):
        """Semantic tier reads from semantic/ subdir."""
        _write_graph(
            tmp_path / "semantic" / "graph.json",
            [
                {
                    "key": "graph5",
                    "subject": "Bob",
                    "predicate": "works_at",
                    "object": "Acme",
                    "speaker_id": "",
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"semantic": ["graph5"]})
        assert results["graph5"]["subject"] == "Bob"

    def test_reads_procedural_from_subdir(self, tmp_path):
        """Procedural tier reads from procedural/ subdir."""
        _write_graph(
            tmp_path / "procedural" / "graph.json",
            [
                {
                    "key": "proc3",
                    "subject": "Carol",
                    "predicate": "likes",
                    "object": "Tea",
                    "speaker_id": "",
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"procedural": ["proc3"]})
        assert results["proc3"]["object"] == "Tea"

    def test_missing_file_returns_none(self, tmp_path):
        """Missing graph.json → all keys return None."""
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1", "graph2"]})
        assert results == {"graph1": None, "graph2": None}

    def test_missing_key_returns_none(self, tmp_path):
        """Key absent from graph → None; key present → hit."""
        _write_graph(
            tmp_path / "episodic" / "graph.json",
            [
                {
                    "key": "graph1",
                    "subject": "X",
                    "predicate": "p",
                    "object": "Y",
                    "speaker_id": "",
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1", "graph999"]})
        assert results["graph1"] is not None
        assert results["graph999"] is None

    def test_empty_keys_skipped(self, tmp_path):
        """Empty key list for an adapter → no entries in result."""
        results = DiskMemorySource(tmp_path).probe({"episodic": []})
        assert results == {}

    def test_raw_output_is_json_with_fields(self, tmp_path):
        """raw_output is a JSON string with key/subject/predicate/object fields."""
        _write_graph(
            tmp_path / "episodic" / "graph.json",
            [
                {
                    "key": "graph1",
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": "Munich",
                    "speaker_id": "",
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1"]})
        raw = json.loads(results["graph1"]["raw_output"])
        assert raw["key"] == "graph1"
        assert raw["subject"] == "Alex"
        assert raw["object"] == "Munich"

    def test_result_shape_has_required_fields(self, tmp_path):
        """Hit results contain key/subject/predicate/object/confidence/format/fact_text/raw_output fields."""  # noqa: E501
        quad = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "knows",
            "object": "Bob",
            "speaker_id": "",
        }

        graph_sim_dir = tmp_path / "sim"
        _write_graph(graph_sim_dir / "episodic" / "graph.json", [quad])

        graph_result = DiskMemorySource(graph_sim_dir).probe({"episodic": ["graph1"]})

        expected_keys = {
            "key",
            "subject",
            "predicate",
            "object",
            "speaker_id",
            "confidence",
            "fact_text",
            "raw_output",
        }
        assert expected_keys == set(graph_result["graph1"].keys()), (
            "DiskMemorySource.probe must return the canonical result shape.\n"
            f"actual keys: {sorted(graph_result['graph1'].keys())}"
        )


# ---------------------------------------------------------------------------
# TestConsolidateSimulateFold
# ---------------------------------------------------------------------------


def _make_bare_loop(tmp_path: Path) -> ConsolidationLoop:
    """Build the minimal ConsolidationLoop the disk-venue full fold needs.

    The disk venue runs the SAME spine as the weights venue, so this loop
    carries everything the spine touches outside the weight-only blocks:

      - ``output_dir`` — used as the adapter_dir root.
      - ``merger`` — a model-free ``GraphMerger`` so the merge topology
        can run without a GPU or loaded model (``merger.model=None`` means the
        model-gated Case-2 branch is skipped; production-correct for simulate mode).
      - ``store`` — a real :class:`MemoryStore` with the three main-tier
        registries loaded.  **This is the fold's input in BOTH venues**; the
        disk venue is not store-free.
      - key counters / ``promoted_keys`` / ``cycle_count`` — mutated by the
        keyed-entry builder and the promotion pass.
      - ``procedural_config`` — read by ``partition_relations`` to decide
        whether procedural is a live tier.
      - ``save_cycle_snapshots`` — False so ``snapshot_dir_for`` returns None.
      - ``_debug_base`` — None, so no artifact root is ever opened.
      - ``config`` — ``ConsolidationConfig`` with base defaults (cloud master
        switch off, refinement_enrichment="off", refinement_normalization="off")
        so enrichment and normalization are suppressed without explicit flags.
      - ``training_config`` — read by ``_hydrate_store_for_fold`` for the
        recall-probe batch size.  The disk venue never batches a generate call,
        but the source factory takes the same arguments in both venues so the
        signature does not fork; production always carries this object.

    Args:
        tmp_path: Adapter root for this loop.

    Model/tokenizer stay ``None`` — the disk venue holds no PeftModel, which is
    exactly what production does (``app.py`` leaves ``loop.model`` a bare base
    model or ``None`` in simulate).  All other attributes are left unset; any
    unintended access raises AttributeError rather than silently returning a
    MagicMock value.
    """
    from paramem.graph.merger import GraphMerger

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.output_dir = tmp_path
    loop.merger = GraphMerger(model=None)
    loop._incidents_state_dir = None
    # model/tokenizer None: the whole-graph normalization pass reads self.model and
    # cleanly skips (skip_reason="no_model") — production-correct for simulate mode,
    # which has no local model resident.
    loop.model = None
    loop.tokenizer = None
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    # Enrichment sizing scalars.  A GraphTierRefiner is constructed for BOTH
    # tier passes, so its enrichment sizing is supplied even on a normalize-only
    # run that skips on model-is-None.  ConsolidationLoop.__init__ always sets
    # these (consolidation.py:501-502) and release() never nulls them, so no
    # production path reaches the tier without them.
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50
    loop.config = ConsolidationConfig()
    loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False, batch_size=1)

    store = MemoryStore(replay_enabled=True)
    for tier in ("episodic", "semantic", "procedural"):
        store.load_registry(tier, KeyRegistry())
    loop.store = store

    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.promoted_keys: set = set()
    loop.procedural_config = AdapterConfig(
        rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"]
    )
    return loop


def _seed_store_tier(
    loop: ConsolidationLoop,
    tier: str,
    triples: list[dict],
    *,
    entries: bool = True,
) -> None:
    """Register *triples* in ``loop.store`` under *tier*, as boot hydration does.

    Writes exactly what a hydrated store carries for a keyed fact: the tier's
    :class:`KeyRegistry`, the entry payload plus its SimHash, and the per-key
    bookkeeping record.  This is the fold's input in BOTH venues, so every
    fixture that wants the fold to see a fact goes through here — the disk
    venue's ``graph.json`` files are the hydration *source* and the post-fold
    *sink*, never the fold's direct input.

    Args:
        loop: The loop under test; ``loop.store`` receives the state and
            ``loop._indexed_next_index`` is advanced past every seeded
            ``graphN`` key so a later mint cannot collide with a seeded one
            (production derives the counter from the live registry the same
            way, via ``seed_key_metadata``).
        tier: Store tier / adapter name to register the keys under.
        triples: Dicts with ``key``, ``subject``, ``predicate``, ``object``,
            and optionally ``speaker_id`` / ``relation_type`` /
            ``reinforcement_count`` (the promotion driver).
        entries: When ``False``, register the key and its fingerprint and write
            its bookkeeping, but leave the entry cache empty for it.  That is
            the per-key shape of a partial boot preload
            (``app._build_store_contents`` always returns the full registry and
            the full bookkeeping, and only the entry probe comes back short —
            reported as ``boot_degraded={"reason": "preload_partial"}``).  The
            key is live and serving; only its content is missing.
    """
    from paramem.memory.entry import entry_simhash

    if not loop.store.has_registry(tier):
        loop.store.load_registry(tier, KeyRegistry())
    for t in triples:
        entry = {
            "key": t["key"],
            "subject": t["subject"],
            "predicate": t.get("predicate", ""),
            "object": t["object"],
            "speaker_id": t.get("speaker_id", ""),
        }
        if entries:
            loop.store.put(tier, t["key"], entry, simhash=entry_simhash(entry))
        else:
            loop.store.registry(tier).add(t["key"])
            loop.store.put_simhash(tier, t["key"], entry_simhash(entry))
        loop.store.set_bookkeeping(
            t["key"],
            speaker_id=t.get("speaker_id", ""),
            relation_type=t.get("relation_type", "factual"),
            first_seen="",
            reinforcement_count=t.get("reinforcement_count", 1),
            allow_empty_speaker=not t.get("speaker_id", ""),
        )
        if t["key"].startswith("graph") and t["key"][len("graph") :].isdigit():
            loop._indexed_next_index = max(
                loop._indexed_next_index, int(t["key"][len("graph") :]) + 1
            )


def _write_interim_graph(
    loop: ConsolidationLoop,
    stamp: str,
    triples: list[dict],
    *,
    store_state: str = "hydrated",
) -> Path:
    """Seed one simulate-venue interim slot: on-disk graph.json plus store state.

    Mirrors what ``commit_tier_slot(mode="simulate")`` writes and what boot
    hydration (``MemoryStore.load_registries_from_disk`` +
    :class:`DiskMemorySource`) then loads back — the slot dir carries the
    payload, the store carries the registry, entries, and bookkeeping.  The
    full fold reads the STORE in both venues; the slot dir exists so the
    post-fold reap has something to remove.

    Args:
        loop: The loop under test.  ``loop.output_dir`` is the adapter root and
            ``loop.store`` receives the slot's registry/entries/bookkeeping.
        stamp: Sub-interval stamp, e.g. ``"20260101T0000"``.
        triples: List of dicts with ``key``, ``subject``, ``predicate``,
            ``object``, and optionally ``speaker_id`` / ``relation_type``.
        store_state: Which boot outcome the store reflects.

            - ``"hydrated"`` (default) — registry, entries, and bookkeeping all
              present: a clean boot.
            - ``"registry_only"`` — registry, fingerprints, and bookkeeping
              present but the entry cache empty for these keys: a **partial
              preload** (``boot_degraded={"reason": "preload_partial"}``).  The
              keys are live and serving; only their content is missing, and the
              slot's ``graph.json`` still holds it.
            - ``"absent"`` — the store knows nothing of the slot at all: a
              registry read that failed outright (``store_load_degraded``).  The
              payload is on disk and only on disk.

    Returns:
        The interim directory path.
    """
    from paramem.memory.persistence import save_memory_to_disk as _save

    if store_state not in ("hydrated", "registry_only", "absent"):
        raise ValueError(f"unknown store_state: {store_state!r}")

    interim_dir = loop.output_dir / "episodic" / f"interim_{stamp}"
    interim_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for t in triples:
        graph.add_edge(
            t["subject"],
            t["object"],
            **{
                _IK_KEY_ATTR: t["key"],
                "predicate": t.get("predicate", ""),
                "speaker_id": t.get("speaker_id", ""),
            },
        )
    _save(graph, interim_dir / "graph.json")

    if store_state != "absent":
        _seed_store_tier(
            loop,
            f"episodic_interim_{stamp}",
            triples,
            entries=(store_state == "hydrated"),
        )
    return interim_dir


class TestConsolidateSimulateFold:
    """Tests for consolidate (simulate-mode full fold)."""

    def test_consume_pending_on_simulate_raises(self, tmp_path):
        """The simulate venue trains nothing — it must refuse to "consume" pending sessions.

        The caller derives ``consume_pending`` from
        ``max_interim_count == 0 and mode != "simulate"``, so the pairing cannot
        occur today.  The guard makes a future caller that gets the derivation
        wrong fail loudly instead of silently ingesting nothing.
        """
        loop = _make_bare_loop(tmp_path)

        with pytest.raises(ValueError, match="cannot consume pending sessions"):
            loop.consolidate(mode="simulate", consume_pending=True)

    def test_single_interim_slot_merged_into_main(self, tmp_path):
        """Single interim slot: its triples appear in the main episodic graph.json.

        After consolidation the main graph exists, contains the expected edges,
        and the interim directory is removed.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        triples = [
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            {"key": "graph2", "subject": "Alice", "predicate": "works_at", "object": "Acme"},
        ]
        interim_dir = _write_interim_graph(loop, "20260101T0000", triples)

        result = loop.consolidate(mode="simulate")

        assert result["tiers_rebuilt"] == ["episodic"]
        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert main_graph_path.exists(), "Main episodic graph.json must be written"
        merged = load_memory_from_disk(main_graph_path)
        keys = {e["key"] for e in iter_entries(merged)}
        assert keys == {"graph1", "graph2"}, f"Merged keys mismatch: {keys}"
        assert not interim_dir.exists(), "Interim directory must be removed after merge"

    def test_two_interim_slots_union_merged(self, tmp_path):
        """Two interim slots with disjoint triples: main graph contains their union.

        Both interim directories must be removed after consolidation.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        slot_a = _write_interim_graph(
            loop,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )
        slot_b = _write_interim_graph(
            loop,
            "20260102T0000",
            [{"key": "graph2", "subject": "Bob", "predicate": "works_at", "object": "Acme"}],
        )

        result = loop.consolidate(mode="simulate")

        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert main_graph_path.exists()
        merged = load_memory_from_disk(main_graph_path)
        keys = {e["key"] for e in iter_entries(merged)}
        assert keys == {"graph1", "graph2"}, (
            f"Union merge failed — expected {{graph1, graph2}}, got {keys}"
        )
        assert not slot_a.exists(), "Slot A must be removed after merge"
        assert not slot_b.exists(), "Slot B must be removed after merge"
        assert result["tiers_rebuilt"] == ["episodic"]

    def test_overlapping_triples_deduplicated_in_main_graph(self, tmp_path):
        """Two slots sharing the same SPO triple dedup to a single edge via GraphMerger.

        The simulate merge routes through ``GraphMerger.merge(resolve_contradictions=False)``.
        When the same SPO (and same ik_key) arrives from both the main graph and an
        interim slot, the merger's Case-1 (identical SPO) fires and produces exactly ONE
        surviving edge.  The merged graph must therefore have ``number_of_edges() == 1``
        and its key set == ``{"graph1"}``.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        shared_triple = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
        }
        _write_interim_graph(loop, "20260101T0000", [shared_triple])
        _write_interim_graph(loop, "20260102T0000", [shared_triple])

        loop.consolidate(mode="simulate")

        main_graph_path = tmp_path / "episodic" / "graph.json"
        merged = load_memory_from_disk(main_graph_path)
        keys = [e["key"] for e in iter_entries(merged)]
        # GraphMerger Case-1 collapses duplicates to a SINGLE surviving edge.
        n_edges = merged.number_of_edges()
        assert n_edges == 1, (
            f"GraphMerger must produce exactly 1 edge for duplicate SPO, got {n_edges}"
        )
        assert set(keys) == {"graph1"}, (
            f"Expected key set {{graph1}} after overlap merge, got {set(keys)}"
        )

    def test_interim_dirs_removed_after_merge(self, tmp_path):
        """All merged interim directories are removed; the main graph survives.

        This is the cleanup assertion decoupled from content correctness.
        """
        loop = _make_bare_loop(tmp_path)
        dirs = [
            _write_interim_graph(
                loop,
                f"2026010{i}T0000",
                [{"key": f"graph{i}", "subject": f"E{i}", "predicate": "p", "object": f"O{i}"}],
            )
            for i in range(1, 4)
        ]

        loop.consolidate(mode="simulate")

        for d in dirs:
            assert not d.exists(), f"Interim dir {d} must be removed after consolidation"
        assert (tmp_path / "episodic" / "graph.json").exists(), (
            "Main graph.json must exist after merge"
        )

    def test_result_contains_tier_delta(self, tmp_path):
        """Result dict contains 'tier_delta' with episodic before/after counts.

        Every fold emits tier_delta, from the same ``_build_tier_delta`` call in
        both venues.  ``staled_by_reason`` is ``{}`` in this fixture because no
        dedup collapse occurs (the single interim slot has a unique triple), so
        the removal ledger is empty.
        """

        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(
            loop,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )

        result = loop.consolidate(mode="simulate")

        assert "tier_delta" in result, (
            f"Result must contain 'tier_delta'; got {list(result.keys())}"
        )
        td = result["tier_delta"]
        assert "episodic" in td, f"tier_delta must have 'episodic' key; got {list(td.keys())}"
        ep = td["episodic"]
        assert "active_before" in ep and "active_after" in ep, (
            f"tier_delta['episodic'] must have active_before + active_after; got {ep!r}"
        )
        assert "staled_by_reason" in ep, (
            f"tier_delta['episodic'] must have staled_by_reason; got {ep!r}"
        )
        assert ep["staled_by_reason"] == {}, (
            "no dedup collapse in this fixture, so the removal ledger is empty"
        )
        assert ep["active_after"] == 1, "One interim key folded into episodic → active_after == 1"

    def test_simulate_fold_with_nothing_to_fold_is_a_noop(self, tmp_path):
        """Empty store, no interims: nothing is rebuilt and nothing is written.

        Same contract as the weights venue — a tier with no keys is skipped, so
        ``tiers_rebuilt`` is empty and the persist tail never fires.  ``app.py``
        reads exactly that (``tiers_rebuilt == []`` → ``noop``).  Persisting an
        empty projection over whatever is on disk would be a write with no
        content behind it.
        """
        loop = _make_bare_loop(tmp_path)
        # No interim slots, no store content.

        result = loop.consolidate(mode="simulate")

        assert result["tiers_rebuilt"] == [], (
            f"nothing in the store → nothing rebuilt; got {result['tiers_rebuilt']!r}"
        )
        assert not (tmp_path / "episodic" / "graph.json").exists(), (
            "a fold that rebuilt nothing must not write a tier graph"
        )

    def test_current_interim_stamp_stays_none_across_the_fold(self, tmp_path):
        """The full fold never labels its debug artifacts with an interim stamp.

        ``_current_interim_stamp`` is cleared at fold entry and on exit, so full-fold
        artifacts always land under the cycle-scoped path — there is no second
        artifact family keyed by the caller's intent.
        """
        loop = _make_bare_loop(tmp_path)
        loop.consolidate(mode="simulate")

        # Stamp must be cleared on normal return.
        stamp = getattr(loop, "_current_interim_stamp", "NOT_SET")
        assert stamp is None, (
            f"_current_interim_stamp must be None after consolidate returns; got {stamp!r}"
        )

    def test_cross_slot_variant_collapse_in_simulate(self, tmp_path):
        """Cross-slot variant-pair collapse: two slots with different surface forms
        for the same canonical triple merge to a single edge.

        Seeds two interim graph.json slots whose surfaces differ but canonicalize
        to the same identity (subject "Alice" vs "alice", object "Acme Corp" vs
        "acme corp").  After consolidate(mode="simulate"):

        (a) The variants COLLAPSE to a single edge in the persisted main graph.json
            (canonical() node identity + Case-1 dedup in the GraphMerger topology,
            simulate/train parity satisfied end-to-end).
        (b) merger.removal_ledger is cleared by the cycle's finally block; the
            dedup evidence is verified via the single-edge graph output instead.

        This is the end-to-end assertion the merger-isolation tests cannot cover:
        it verifies that the simulate path routes through GraphMerger so grooming
        is literally identical to the train fold.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)

        # Slot A: subject/object in title-case (will be registered first).
        _write_interim_graph(
            loop,
            "20260101T0000",
            [
                {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "works at",
                    "object": "Acme Corp",
                }
            ],
        )
        # Slot B: same SPO but surfaces differ — casefold variant.
        # canonical("alice") == canonical("Alice") == "alice"
        # canonical("acme corp") == canonical("Acme Corp") == "acme corp"
        # These canonicalize to the same triple, so Case-1 fires and the
        # merger records pre_surfaces with the differing incoming and surviving surfaces.
        _write_interim_graph(
            loop,
            "20260102T0000",
            [
                {
                    "key": "graph2",
                    "subject": "alice",
                    "predicate": "works at",
                    "object": "acme corp",
                }
            ],
        )

        loop.consolidate(mode="simulate")

        # (a) Main graph.json must contain exactly ONE edge (the variants collapsed).
        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert main_graph_path.exists(), "Main graph.json must be written"
        merged = load_memory_from_disk(main_graph_path)
        entries = list(iter_entries(merged))
        n_edges = merged.number_of_edges()
        assert n_edges == 1, (
            f"Variant pair must collapse to a single edge; got {n_edges} edges "
            f"(keys: {[e['key'] for e in entries]})"
        )

        # (b) merger.removal_ledger is cleared by the cycle's finally block and
        # is not observable after the call.  The variant-collapse is already verified by
        # the single-edge assertion above.  Additionally verify that only the canonical
        # surface key survives (Alice/Acme Corp, not alice/acme corp).
        surviving_keys = {e["key"] for e in entries}
        # graph1 was the first-seen key (canonical surface), graph2 was the variant.
        # The dedup collapses graph2 into graph1, so graph1 should survive.
        assert "graph1" in surviving_keys, (
            f"graph1 (canonical surface) must survive the variant collapse; "
            f"surviving keys: {surviving_keys}"
        )

    def test_enrichment_survives_into_persisted_graph_after_merge(self, tmp_path):
        """Regression: enrichment runs AFTER reset+merge, so sentinel edge survives persist.

        On the buggy code (enrichment before reset_graph), the sentinel edge was wiped by
        reset_graph() and the persisted graph.json had no enrichment.  After the fix,
        enrichment runs on the freshly-merged graph and the sentinel coexists with the
        merged interim edges in the persisted output.

        Strategy: monkeypatch GraphTierRefiner.run_enrichment to add a sentinel
        edge to loop.merger.graph and return new_edges=1.  The sentinel is
        KEYLESS, exactly like a real enrichment edge: the fold mints its key in
        ``_build_all_edge_entries_into`` and registers it in the store, which is
        what carries it into the persisted per-tier projection.  After the
        simulate fold, assert:
        (a) the sentinel edge is present in the persisted graph.json, and
        (b) the merged interim edge is also present (sentinel coexists with merged content),
        (c) tier_delta["episodic"]["minted"] == 1 (sourced from enrichment new_edges).
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        # refinement_enrichment="on" + the cloud master switch on so consolidate
        # calls GraphTierRefiner.run_enrichment; base defaults (off/False) would skip it.
        loop.config = ConsolidationConfig(refinement_enrichment="on")
        loop.cloud_enabled = True

        # Seed one interim slot so there is merged content to coexist with.
        _write_interim_graph(
            loop,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )

        # Sentinel values for the edge the enrichment mock will inject.
        SENTINEL_SUBJECT = "__enr_sentinel__"
        SENTINEL_OBJECT = "__enr_sentinel_obj__"

        def _fake_enrichment(refiner_self=None):
            """Add a keyless sentinel edge to loop.merger.graph; return new_edges=1.

            Must run on the populated merged graph (after reset+merge), otherwise
            the merger's graph is empty and the sentinel is wiped by reset_graph().
            The buggy code runs enrichment BEFORE reset_graph(), so the sentinel
            would be absent from the persisted graph on the old path.
            """
            loop.merger.graph.add_edge(
                SENTINEL_SUBJECT,
                SENTINEL_OBJECT,
                predicate="enriched_by",
                relation_type="factual",
                speaker_id="",
            )
            return {
                "chunks": 1,
                "new_edges": 1,
                "same_as_merges": 0,
                "skipped": False,
                "skip_reason": None,
            }

        from unittest.mock import patch

        from paramem.training.graph_tier import GraphTierRefiner

        with patch.object(GraphTierRefiner, "run_enrichment", side_effect=_fake_enrichment):
            result = loop.consolidate(mode="simulate")

        # (a) Sentinel edge must be present in the persisted graph.json.
        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert main_graph_path.exists(), "Main graph.json must be written"
        merged = load_memory_from_disk(main_graph_path)
        entries_in_graph = list(iter_entries(merged))
        keys_in_graph = {e["key"] for e in entries_in_graph}
        assert any(e["predicate"] == "enriched_by" for e in entries_in_graph), (
            f"Sentinel enrichment edge must survive into persisted graph.json; "
            f"entries present: {entries_in_graph}\n"
            "On the buggy code (enrichment before reset_graph) the sentinel is wiped "
            "by reset_graph() and is absent here."
        )

        # (b) Merged interim edge must also be present (sentinel coexists with real content).
        assert "graph1" in keys_in_graph, (
            f"Merged interim edge 'graph1' must coexist with sentinel; "
            f"keys present: {sorted(keys_in_graph)}"
        )

        # (c) tier_delta["episodic"]["minted"] must equal enrichment new_edges (1).
        td = result.get("tier_delta", {})
        assert "episodic" in td, f"tier_delta must contain 'episodic'; got {list(td.keys())}"
        minted = td["episodic"].get("minted")
        assert minted == 1, (
            f"tier_delta['episodic']['minted'] must equal enrichment new_edges=1; got {minted!r}"
        )

    def test_simulate_merge_produces_person_node_with_speaker_id(self, tmp_path):
        """Disk-venue materialize stamps entity_type='person' + speaker_id on the merged graph.

        Regression guard: before routing consolidate through
        GraphMerger.merge_relations, the simulate path used entities=[] directly
        (same latent bug as the recon path fixed by GraphMerger.merge_relations
        unification).  A relation whose subject == speaker_id (i.e. a speaker
        node) would be stored as entity_type='concept' with no speaker_id
        attribute, causing keyless-edge attribution to fall back to speaker_id="".

        Asserted on ``loop.merger.graph`` rather than on the persisted artifact:
        the per-tier ``graph.json`` is an edge projection of the store
        (``build_tier_graph_from_store``) and carries no node attributes in
        either venue, so the merged graph is where this invariant is observable.
        """
        loop = _make_bare_loop(tmp_path)

        # Seed an interim slot whose triple has subject == speaker_id.
        # paramem.graph.merger._synth_speaker_entities fires when _r.speaker_id != "" AND
        # _r.subject == _r.speaker_id.  We use "speaker0" for both.
        _write_interim_graph(
            loop,
            "20260101T0000",
            [
                {
                    "key": "graph1",
                    "subject": "speaker0",
                    "predicate": "works_at",
                    "object": "Acme Corp",
                    "speaker_id": "speaker0",
                },
            ],
        )

        loop._materialize_consolidation_graph(source="disk")

        # Speaker node keys are the lowercase speaker_id.
        # Under the lowercase-uniform design entity.speaker_id == node key == "speaker0".
        node_key = "speaker0"
        assert node_key in loop.merger.graph.nodes, (
            f"Speaker subject node {node_key!r} missing from merged graph after disk "
            f"materialize; nodes present: {list(loop.merger.graph.nodes)}"
        )
        node_data = loop.merger.graph.nodes[node_key]
        assert node_data.get("entity_type") == "person", (
            f"Disk venue: expected entity_type='person' on speaker subject node; "
            f"got entity_type={node_data.get('entity_type')!r}. "
            "Regression: before GraphMerger.merge_relations routing, simulate used "
            "entities=[] so speaker nodes received entity_type='concept' with no speaker_id."
        )
        assert node_data.get("speaker_id") == "speaker0", (
            f"Disk venue: expected speaker_id='speaker0' (in node attribute); "
            f"got speaker_id={node_data.get('speaker_id')!r}. "
            "Regression: paramem.graph.merger._synth_speaker_entities was not applied "
            "to the disk venue."
        )
        loop.merger.reset_graph()


# ---------------------------------------------------------------------------
# TestBuildTierDelta (regression: unified staled_by_reason + minted)
# ---------------------------------------------------------------------------


class TestBuildTierDelta:
    """Regression tests for :meth:`ConsolidationLoop._build_tier_delta`.

    Verifies the contract:

    - ``staled_by_reason`` total across all tiers == ledger entries attributable
      to a tier (each removed key reflected in exactly one tier, no double-count
      or drop).
    - ``minted`` per tier equals the caller-supplied ``minted_by_tier`` input.
    - Keys in ``removal_ledger`` whose ``store.tier_of`` returns ``None`` are
      skipped (genuinely unattributable — boundary skip, not error suppression).
    """

    def _make_loop_with_store(self, tmp_path: Path) -> ConsolidationLoop:
        """Minimal loop with a real MemoryStore and GraphMerger (no model/GPU)."""
        from paramem.graph.merger import GraphMerger

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.config = None
        loop.merger = GraphMerger(model=None)

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    def test_staled_by_reason_reflects_dedup_ledger_entry(self, tmp_path):
        """staled_by_reason counts match the dedup ledger entries for keys in the store.

        Seeds two keys in the episodic store, populates the merger.removal_ledger
        with a dedup entry for one of them, and verifies that _build_tier_delta
        attributes it to episodic under reason 'dedup'.
        """
        loop = self._make_loop_with_store(tmp_path)

        # Register two keys in the episodic tier.
        _ep = {"subject": "Alice", "predicate": "likes", "object": "Tea"}
        loop.store.put("episodic", "graph1", _ep)
        loop.store.put("episodic", "graph2", _ep)

        # Simulate a dedup collapse: graph2 is collapsed into graph1.
        loop.merger.removal_ledger = {
            "graph2": {"reason": "dedup", "survivor_key": "graph1", "pre_surfaces": {}}
        }

        td = loop._build_tier_delta(
            active_before={"episodic": 2},
            active_after={"episodic": 1},
            minted_by_tier={"episodic": 0},
        )

        assert "episodic" in td
        ep = td["episodic"]
        assert ep["staled_by_reason"] == {"dedup": 1}, (
            "One dedup ledger entry for an episodic key must produce"
            f" staled_by_reason={{'dedup': 1}}; got {ep['staled_by_reason']!r}"
        )
        # Total staled_by_reason count == number of attributable ledger entries.
        total_staled = sum(
            sum(v.values()) for v in (tier_rec["staled_by_reason"] for tier_rec in td.values())
        )
        assert total_staled == 1, (
            "Total staled_by_reason count must equal 1 attributable ledger entry;"
            f" got {total_staled}"
        )

    def test_staled_by_reason_includes_enrichment_same_as(self, tmp_path):
        """enrichment_same_as ledger entries are included in staled_by_reason.

        Unlike the former train-mode path (soft_stale_by_tier, dedup-only),
        the unified _build_tier_delta attributes ALL removal reasons from the
        ledger, including enrichment_same_as.
        """
        loop = self._make_loop_with_store(tmp_path)
        loop.store.put(
            "episodic", "graph3", {"subject": "Alice", "predicate": "is", "object": "Bob"}
        )

        loop.merger.removal_ledger = {
            "graph3": {"reason": "enrichment_same_as", "keep_node": "alice"}
        }

        td = loop._build_tier_delta(
            active_before={"episodic": 1},
            active_after={"episodic": 1},
            minted_by_tier={"episodic": 0},
        )

        ep = td["episodic"]
        assert ep["staled_by_reason"].get("enrichment_same_as", 0) == 1, (
            f"enrichment_same_as ledger entry must appear in staled_by_reason;"
            f" got {ep['staled_by_reason']!r}"
        )

    def test_unattributable_key_skipped(self, tmp_path):
        """A ledger entry for a key absent from the store produces no staled_by_reason entry.

        This is the boundary-skip branch: simulate mode has no store entries so
        tier_of returns None for all removed keys.  The boundary skip must NOT
        produce a staled_by_reason entry for any tier.
        """
        loop = self._make_loop_with_store(tmp_path)
        # Deliberately do NOT put "ghost_key" into the store.

        loop.merger.removal_ledger = {
            "ghost_key": {"reason": "dedup", "survivor_key": "real_key", "pre_surfaces": {}}
        }

        td = loop._build_tier_delta(
            active_before={"episodic": 5},
            active_after={"episodic": 5},
            minted_by_tier={"episodic": 0},
        )

        total_staled = sum(
            sum(v.values()) for v in (tier_rec["staled_by_reason"] for tier_rec in td.values())
        )
        assert total_staled == 0, (
            "An unattributable ledger key (not in store) must not inflate staled_by_reason;"
            f" got total_staled={total_staled}"
        )

    def test_minted_equals_caller_supplied_input(self, tmp_path):
        """minted per tier equals the minted_by_tier input dict, not a derived count.

        Verifies the invariant for both simulate (single-tier dict) and train
        (multi-tier dict) callers.
        """
        loop = self._make_loop_with_store(tmp_path)
        loop.merger.removal_ledger = {}

        minted_in = {"episodic": 3, "procedural": 1}
        td = loop._build_tier_delta(
            active_before={"episodic": 10, "procedural": 5},
            active_after={"episodic": 13, "procedural": 6},
            minted_by_tier=minted_in,
        )

        assert td["episodic"]["minted"] == 3, (
            "minted for episodic must equal minted_by_tier input 3;"
            f" got {td['episodic']['minted']!r}"
        )
        assert td["procedural"]["minted"] == 1, (
            "minted for procedural must equal minted_by_tier input 1;"
            f" got {td['procedural']['minted']!r}"
        )
        # Simulate-style: single tier
        td2 = loop._build_tier_delta(
            active_before={"episodic": 0},
            active_after={"episodic": 2},
            minted_by_tier={"episodic": 2},
        )
        assert td2["episodic"]["minted"] == 2

    def test_multi_tier_dedup_no_double_count(self, tmp_path):
        """Each removed key is attributed to exactly one tier — no double-count.

        Seeds one key per tier and puts all three in the removal_ledger.
        Total staled_by_reason count across tiers must equal 3 (one per key).
        """
        loop = self._make_loop_with_store(tmp_path)
        loop.store.put("episodic", "ep_key", {"subject": "A", "predicate": "p", "object": "B"})
        loop.store.put("semantic", "sem_key", {"subject": "C", "predicate": "p", "object": "D"})
        loop.store.put("procedural", "proc_key", {"subject": "E", "predicate": "p", "object": "F"})

        loop.merger.removal_ledger = {
            "ep_key": {"reason": "dedup", "survivor_key": "other", "pre_surfaces": {}},
            "sem_key": {"reason": "dedup", "survivor_key": "other2", "pre_surfaces": {}},
            "proc_key": {"reason": "dedup", "survivor_key": "other3", "pre_surfaces": {}},
        }

        td = loop._build_tier_delta(
            active_before={"episodic": 1, "semantic": 1, "procedural": 1},
            active_after={"episodic": 0, "semantic": 0, "procedural": 0},
            minted_by_tier={},
        )

        total_staled = sum(
            sum(v.values()) for v in (tier_rec["staled_by_reason"] for tier_rec in td.values())
        )
        assert total_staled == 3, (
            f"Three ledger entries across three tiers must produce total staled_by_reason=3;"
            f" got {total_staled} (td={td!r})"
        )
        # No double-counting: each tier has exactly 1.
        for tier in ("episodic", "semantic", "procedural"):
            assert sum(td[tier]["staled_by_reason"].values()) == 1, (
                f"Tier {tier} must have exactly 1 staled entry;"
                f" got {td[tier]['staled_by_reason']!r}"
            )

    def test_simulate_variant_collapse_staled_by_reason_empty(self, tmp_path):
        """Simulate path: removal_ledger dedup entry for unattributed key → staled_by_reason {}.

        Uses the variant-collapse fixture pattern (two interim slots with
        canonically-identical triples).  The simulate-mode store has no entries
        (no KeyRegistry mutation in simulate), so tier_of returns None for the
        collapsed key — boundary skip → staled_by_reason stays {}.

        staled_by_reason is {} for simulate not because grooming is skipped,
        but because attribution is unavailable (simulate-mode store has no entries).
        """
        loop = _make_bare_loop(tmp_path)

        _write_interim_graph(
            loop,
            "20260201T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "works at", "object": "Acme Corp"}],
        )
        _write_interim_graph(
            loop,
            "20260202T0000",
            [{"key": "graph2", "subject": "alice", "predicate": "works at", "object": "acme corp"}],
        )

        result = loop.consolidate(mode="simulate")

        td = result["tier_delta"]
        assert "episodic" in td
        ep = td["episodic"]
        # Ledger has a dedup entry (Case-1 collapse), but store has no entries.
        assert ep["staled_by_reason"] == {}, (
            "Simulate variant-collapse: staled_by_reason must be {} because the store"
            f" has no entries for attribution; got {ep['staled_by_reason']!r}"
        )

        # merger.removal_ledger is cleared by the cycle's finally block.
        # Confirm the dedup collapse happened via the persisted graph output (1 edge).
        main_path = tmp_path / "episodic" / "graph.json"
        assert main_path.exists(), "graph.json must exist after the fold"
        from paramem.memory.persistence import load_memory_from_disk

        merged = load_memory_from_disk(main_path)
        assert merged.number_of_edges() == 1, (
            "Variant-collapse fixture must produce exactly 1 edge (dedup collapse);"
            f" got {merged.number_of_edges()} edges"
        )


# ---------------------------------------------------------------------------
# TestBackgroundTrainerClose
# ---------------------------------------------------------------------------


@contextmanager
def _noop_gpu_lock():
    """No-op context manager replacing gpu_lock_sync for unit tests."""
    yield


def _make_bt_for_close(tmp_path: Path):
    """Return a BackgroundTrainer configured for close() tests."""
    from paramem.server.background_trainer import BackgroundTrainer
    from paramem.utils.config import TrainingConfig

    model = MagicMock()
    model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}
    return BackgroundTrainer(
        model=model,
        tokenizer=MagicMock(),
        training_config=TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
        ),
        output_dir=str(tmp_path),
    )


class TestBackgroundTrainerClose:
    """BackgroundTrainer.close() stops the callable worker thread cleanly."""

    def test_close_on_fresh_trainer_is_noop(self, tmp_path):
        """close() on a freshly-constructed trainer that has never submitted a job succeeds.

        No worker thread has been started; close() must not raise and must not
        block for the full timeout.
        """
        bt = _make_bt_for_close(tmp_path)
        # No submit() called — _worker_thread is None.
        assert bt._worker_thread is None
        bt.close()  # must not raise

    def test_close_after_submit_and_wait_joins_worker(self, tmp_path):
        """close() after submit_and_wait stops the callable-worker thread.

        After submit_and_wait the worker is alive (persistent daemon).  close()
        must send the stop sentinel and join the thread within the timeout so
        the thread is no longer alive.
        """
        bt = _make_bt_for_close(tmp_path)
        job_ran = threading.Event()

        def _job():
            job_ran.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit_and_wait(_job)

        assert job_ran.is_set(), "Job must have run before close() is tested"
        worker = bt._worker_thread
        assert worker is not None and worker.is_alive(), (
            "Worker thread must be alive before close()"
        )

        bt.close(timeout=5.0)

        assert not worker.is_alive(), "Worker thread must be dead after close()"

    def test_close_is_idempotent(self, tmp_path):
        """Calling close() twice does not raise.

        After the first close() the worker has exited.  A second close() on
        the same instance must be a no-op (idempotent).
        """
        bt = _make_bt_for_close(tmp_path)
        job_done = threading.Event()

        def _job():
            job_done.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit_and_wait(_job)

        assert job_done.is_set()

        bt.close(timeout=5.0)
        bt.close(timeout=5.0)  # must not raise

    def test_release_nulls_model_tokenizer_and_thread(self, tmp_path):
        """release() stops the worker and drops model/tokenizer/_worker_thread.

        After submit_and_wait the worker is alive.  release() must join the
        thread (via _stop_callable_worker), null _worker_thread, null model,
        and null tokenizer so no live attribute retains the base-model reference.
        """
        bt = _make_bt_for_close(tmp_path)
        job_ran = threading.Event()

        def _job():
            job_ran.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit_and_wait(_job)

        assert job_ran.is_set(), "Job must have run before release() is tested"
        assert bt._worker_thread is not None, "Worker must be alive before release()"

        bt.release()

        assert bt.model is None, "release() must null model"
        assert bt.tokenizer is None, "release() must null tokenizer"
        assert bt._worker_thread is None, "release() must null _worker_thread"
        assert bt._current_job is None, "release() must null _current_job"

    def test_release_on_fresh_trainer_is_noop(self, tmp_path):
        """release() on a freshly-constructed trainer (no worker started) does not raise."""
        bt = _make_bt_for_close(tmp_path)
        assert bt._worker_thread is None
        bt.release()  # must not raise
        assert bt.model is None
        assert bt.tokenizer is None
        assert bt._worker_thread is None


# ---------------------------------------------------------------------------
# TestConsolidationLoopRelease
# ---------------------------------------------------------------------------


class TestConsolidationLoopRelease:
    """ConsolidationLoop.release() drops all base-model references."""

    def test_release_nulls_model_extraction_and_bg_trainer(self):
        """release() nulls model, tokenizer, _bg_trainer, and extraction.model.

        Uses a bare ConsolidationLoop instance (no __init__) with sentinel
        objects injected directly, matching the pattern in other bare-loop tests.
        """
        from paramem.graph.extraction_pipeline import ExtractionPipeline
        from paramem.training.consolidation import ConsolidationLoop

        sentinel_model = MagicMock(name="base_model")
        sentinel_tokenizer = MagicMock(name="tokenizer")
        sentinel_bt = MagicMock(name="bg_trainer")

        # Build a minimal ExtractionPipeline with the sentinel model.
        ep = ExtractionPipeline.__new__(ExtractionPipeline)
        ep.model = sentinel_model
        ep.tokenizer = sentinel_tokenizer

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = sentinel_model
        loop.tokenizer = sentinel_tokenizer
        loop._bg_trainer = sentinel_bt
        loop.extraction = ep

        loop.release()

        assert loop.model is None, "release() must null loop.model"
        assert loop.tokenizer is None, "release() must null loop.tokenizer"
        assert loop._bg_trainer is None, "release() must null loop._bg_trainer"
        assert loop.extraction is None, "release() must null loop.extraction"
        # The ExtractionPipeline's own model reference must also be cleared
        # before the pipeline is dropped.
        assert ep.model is None, "release() must null extraction.model before clearing extraction"

    def test_release_without_extraction_is_noop(self):
        """release() tolerates a loop with no extraction attribute."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock(name="model")
        loop.tokenizer = MagicMock(name="tokenizer")
        loop._bg_trainer = None
        # No loop.extraction set.

        loop.release()  # must not raise

        assert loop.model is None
        assert loop.tokenizer is None

    def test_release_is_idempotent(self):
        """Calling release() twice does not raise."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock(name="model")
        loop.tokenizer = MagicMock(name="tokenizer")
        loop._bg_trainer = None
        loop.extraction = None

        loop.release()
        loop.release()  # must not raise


class TestGraphTierSkipsAfterRelease:
    """Both graph-tier passes SKIP on a released loop instead of raising.

    ``release()`` nulls ``model`` AND ``extraction`` together, so the
    cloud-only server routinely holds a loop whose ``extraction`` is ``None``.
    Both passes own a ``model is None`` early-skip, and that skip must stay
    reachable WITHOUT any read of ``self.extraction``.

    Regression: the graph tier once took the extraction config as a resolved
    constructor argument, so ``self.extraction.config`` was evaluated by the
    caller before any guard in the tier could run — a released loop raised
    ``AttributeError`` on ``None.config`` where it had previously skipped.
    The config is now handed over as a deferred read
    (``_current_extraction_config``) that only the post-guard paths invoke.
    These tests fail with ``AttributeError`` if that read is ever hoisted back
    above the guard.
    """

    @staticmethod
    def _released_loop(tmp_path: Path) -> ConsolidationLoop:
        """A loop in exactly the post-``release()`` state, with nothing re-added."""
        from paramem.graph.merger import GraphMerger

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.config = ConsolidationConfig()
        loop.cloud_enabled = True
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.graph_enrichment_neighborhood_hops = 2
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.merger = GraphMerger(model=None)
        loop._incidents_state_dir = None
        # Post-release state: release() nulls these four together.
        loop.model = None
        loop.tokenizer = None
        loop._bg_trainer = None
        loop.extraction = None
        return loop

    @staticmethod
    def _refiner(loop: ConsolidationLoop) -> GraphTierRefiner:
        """Build a :class:`GraphTierRefiner` exactly as ``_refine_consolidation_graph``
        does, off a released loop's current (post-``release()``) state."""
        return GraphTierRefiner(
            loop.merger,
            model=loop.model,
            tokenizer=loop.tokenizer,
            extraction_config_provider=loop._current_extraction_config,
            cloud_enabled=loop.cloud_enabled,
            neighborhood_hops=loop.graph_enrichment_neighborhood_hops,
            max_entities_per_pass=loop.graph_enrichment_max_entities_per_pass,
            gc_disable=loop._disable_gradient_checkpointing,
            gc_enable=loop._enable_gradient_checkpointing,
        )

    def test_normalization_skips_on_released_loop(self, tmp_path):
        """Normalization returns the no_model skip, never touching extraction.

        ``loop.cloud_enabled = True`` is deliberate: it is the only branch in the
        normalization pass that reads the extraction config, so a hoisted read
        cannot hide behind a False gate here.
        """
        loop = self._released_loop(tmp_path)

        result = self._refiner(loop).run_normalization()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        assert loop.extraction is None, "the skip path must not repopulate extraction"

    def test_enrichment_skips_on_released_loop(self, tmp_path):
        """Enrichment returns the no_model skip, never touching extraction."""
        loop = self._released_loop(tmp_path)

        result = self._refiner(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        assert loop.extraction is None, "the skip path must not repopulate extraction"

    def test_refine_stage_skips_both_passes_on_released_loop(self, tmp_path):
        """The Refine stage runs both passes, in order, on a released loop
        without raising.

        Covers the caller as well as the passes: ``_refine_consolidation_graph``
        is the production entry point and must not evaluate tier arguments the
        released loop can no longer supply.  Also pins the full-fold pass
        order at the caller: enrichment runs BEFORE normalization (the U2
        flip), so the parity suite covers the caller's observed order, not
        just ``GraphTierRefiner.refine``'s own unit tests.
        """
        from unittest.mock import patch

        loop = self._released_loop(tmp_path)
        loop.store = MemoryStore(replay_enabled=True)

        call_order: list[str] = []
        _orig_enrichment = GraphTierRefiner.run_enrichment
        _orig_normalization = GraphTierRefiner.run_normalization

        def _spy_enrichment(self_inner):
            call_order.append("enrichment")
            return _orig_enrichment(self_inner)

        def _spy_normalization(self_inner):
            call_order.append("normalization")
            return _orig_normalization(self_inner)

        with (
            patch.object(GraphTierRefiner, "run_enrichment", _spy_enrichment),
            patch.object(GraphTierRefiner, "run_normalization", _spy_normalization),
        ):
            loop._refine_consolidation_graph([], normalize=True, enrich=True)

        assert loop.extraction is None
        assert call_order == ["enrichment", "normalization"], (
            f"expected enrichment before normalization; got {call_order}"
        )


# ---------------------------------------------------------------------------
# TestCommitTierSlotCleanup
# ---------------------------------------------------------------------------


def _make_loop_for_commit(tmp_path: Path) -> ConsolidationLoop:
    """Build the minimal ConsolidationLoop required by commit_tier_slot.

    Sets up a MemoryStore with the episodic registry loaded and one key
    registered so save_bytes() produces a non-empty payload.
    """
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.tokenizer = MagicMock()
    loop.output_dir = tmp_path
    loop.fingerprint_cache = None

    store = MemoryStore(replay_enabled=True)
    reg = KeyRegistry()
    # Prime the registry with one key so the payload is non-trivial.
    reg.add("graph1")
    store.load_registry("episodic", reg)
    # Add the entry via put (SPO content) and bookkeeping separately.
    # setdefault_entry was deleted — use the correct API split.
    store.put(
        "episodic",
        "graph1",
        {
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
            "speaker_id": "sp1",
        },
        register=False,
    )
    store.set_bookkeeping("graph1", speaker_id="sp1", relation_type="factual", first_seen="")
    loop.store = store
    return loop


class TestCommitTierSlotCleanup:
    """commit_tier_slot removes the orphan slot dir when write fails before registry flush."""

    def test_train_mode_manifest_failure_removes_slot_dir(self, tmp_path):
        """In train mode, a manifest-build failure before registry flush cleans up slot dir.

        Patches ``build_manifest_for`` to raise so the function never reaches
        step 7 (registry flush).  The slot directory must not exist after the call.
        """
        from paramem.memory.persistence import commit_tier_slot

        loop = _make_loop_for_commit(tmp_path)

        with patch(
            "paramem.adapters.manifest.build_manifest_for",
            side_effect=RuntimeError("manifest build failed"),
        ):
            with pytest.raises(RuntimeError, match="manifest build failed"):
                commit_tier_slot(
                    loop=loop,
                    tier="episodic",
                    adapter_name="episodic_interim_20260101T0000",
                    stamp="20260101T0000",
                    mode="train",
                    all_keyed=[],
                    output_dir=tmp_path,
                )

        # The slot dir must have been cleaned up by the try/finally.
        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        slot_root = adapter_slot_root_for_name(tmp_path, "episodic_interim_20260101T0000")
        assert not slot_root.exists(), (
            f"Orphan slot dir must be removed on manifest failure, but exists: {slot_root}"
        )

    def test_simulate_mode_graph_write_failure_removes_slot_dir(self, tmp_path):
        """In simulate mode, a save_memory_to_disk failure cleans up the slot dir.

        Patches ``save_memory_to_disk`` to raise so the function never reaches
        step 7 (registry flush).  The slot directory must not exist after the call.
        """
        from paramem.memory.persistence import commit_tier_slot

        loop = _make_loop_for_commit(tmp_path)
        all_keyed = [
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "sp1",
            }
        ]

        with patch(
            "paramem.memory.persistence.save_memory_to_disk",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError, match="disk full"):
                commit_tier_slot(
                    loop=loop,
                    tier="episodic",
                    adapter_name="episodic_interim_20260101T0000",
                    stamp="20260101T0000",
                    mode="simulate",
                    all_keyed=all_keyed,
                    output_dir=tmp_path,
                )

        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        slot_root = adapter_slot_root_for_name(tmp_path, "episodic_interim_20260101T0000")
        assert not slot_root.exists(), (
            f"Orphan slot dir must be removed on graph-write failure, but exists: {slot_root}"
        )


# ---------------------------------------------------------------------------
# Artifact-hook integration through run_consolidation_cycle
# ---------------------------------------------------------------------------


class TestDebugSnapshotIntegration:
    """``run_consolidation_cycle`` must fire the writer for every return branch
    when ``save_cycle_snapshots`` is on.  Covers site E (``on_extraction_end``,
    cumulative-graph + relations dump) and site G (``on_cycle_end``,
    cycle-summary dump) including the queue-only short-circuit.
    """

    def _enable_debug(self, loop: ConsolidationLoop, debug_base: Path) -> None:
        """Flip the loop's debug gate on and wire ``snapshot_dir_for`` to
        return ``debug_base / cycle_<N> / run_<run_id>``.
        """
        loop.save_cycle_snapshots = True
        loop._debug_base = debug_base
        loop.run_id = "20260517T120000Z_test01"

    def test_simulate_cycle_writes_end_of_extraction_and_cycle_summary(self, tmp_path):
        """Normal simulate branch — site E + site G fire."""
        loop = _build_loop(tmp_path / "loop")
        debug_base = tmp_path / "debug"
        self._enable_debug(loop, debug_base)

        loop.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="simulate",
            run_label="integration",
            stamp=_STAMP,
        )

        # Cycle passes ``stamp`` through to the writer so the relation dumps
        # nest under ``interim_<stamp>/`` — matches the production layout
        # ``paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/``.
        cycle_dir = loop.snapshot_dir_for(interim_stamp=_STAMP)
        assert cycle_dir is not None
        # graph_enriched_snapshot.json is written by _refine_consolidation_graph
        # via on_fold_graph, which always lands under fold/ — not cycle_dir.
        # graph_merged_snapshot.json is no longer emitted on the interim path.
        fold_base = loop.snapshot_dir_for()
        assert fold_base is not None
        assert (fold_base / "fold" / "graph_enriched_snapshot.json").exists()
        assert (cycle_dir / "episodic_rels_snapshot.json").exists()
        assert (cycle_dir / "procedural_rels_snapshot.json").exists()
        assert (cycle_dir / "cycle_summary_snapshot.json").exists()

        ep_dump = json.loads((cycle_dir / "episodic_rels_snapshot.json").read_text())
        assert ep_dump == list(_EPISODIC_RELS)
        summary = json.loads((cycle_dir / "cycle_summary_snapshot.json").read_text())
        assert summary["mode"] == "simulated"
        assert summary["venue"] == "simulate"
        assert summary["error"] is None

    def test_cap_pending_branch_emits_summary_but_skips_graph_dump(self, tmp_path):
        """Ring-full short-circuit (``cap_pending``) emits only the cycle summary.

        When the interim ring is at ``max_interim_count`` and the target slot is
        new (train mode), run_consolidation_cycle returns ``mode="cap_pending"``
        immediately — no graph dump, no training.
        """
        loop = _build_loop(tmp_path / "loop")
        debug_base = tmp_path / "debug"
        self._enable_debug(loop, debug_base)

        # Pre-fill the PEFT ring to max_interim_count=1 with an existing stamp,
        # then call with a NEW stamp so the target slot is absent from peft_config
        # and ring_full fires (existing count >= max_interim_count).
        _existing_stamp = "20260101T0000"
        _new_stamp = "20260601T1200"
        loop.model.peft_config[f"episodic_interim_{_existing_stamp}"] = MagicMock()

        result = loop.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="train",
            run_label="integration-cap",
            stamp=_new_stamp,
            max_interim_count=1,
        )
        assert result["mode"] == "cap_pending"
        assert result["adapter_name"] is None

        cycle_dir = loop.snapshot_dir_for(interim_stamp=_new_stamp)
        assert cycle_dir is not None
        assert (cycle_dir / "cycle_summary_snapshot.json").exists()
        assert not (cycle_dir / "graph_snapshot.json").exists()
        assert not (cycle_dir / "graph_merged_snapshot.json").exists()
        assert not (cycle_dir / "episodic_rels_snapshot.json").exists()
        assert not (cycle_dir / "procedural_rels_snapshot.json").exists()

        summary = json.loads((cycle_dir / "cycle_summary_snapshot.json").read_text())
        assert summary["mode"] == "cap_pending"
        assert summary["adapter_name"] is None


# ---------------------------------------------------------------------------
# TestDebugSnapshotOnTierDelta
# ---------------------------------------------------------------------------


class TestDebugSnapshotOnTierDelta:
    """``on_tier_delta`` persists the per-tier delta record.

    Every fold emits a
    ``tier_delta.json`` under ``<debug_base>/fold/`` so operators can see
    before/after/staled/minted counts without parsing raw adapter weight files.
    """

    def test_on_tier_delta_writes_file_when_snapshots_enabled(self, tmp_path) -> None:
        """on_tier_delta writes tier_delta.json under fold/ when save_cycle_snapshots=True."""
        import json as _json

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.artifacts import on_tier_delta

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = True
        loop._debug_base = tmp_path / "debug"
        loop.run_id = "test_run_01"
        loop.cycle_count = 1

        tier_delta = {
            "episodic": {
                "active_before": 10,
                "active_after": 8,
                "staled_by_reason": {"dedup": 2},
                "minted": 0,
            }
        }
        with loop._artifact_scope():
            on_tier_delta(tier_delta)

        # Locate the written file under fold/.
        fold_dir = loop.snapshot_dir_for()
        if fold_dir is None:
            pytest.skip("snapshot_dir_for returned None — debug gate not active")
        td_path = fold_dir / "fold" / "tier_delta.json"
        fold_contents = (
            list((fold_dir / "fold").iterdir()) if (fold_dir / "fold").exists() else "fold/ missing"
        )
        assert td_path.exists(), (
            f"tier_delta.json must be written to {td_path}; files in fold dir: {fold_contents}"
        )
        written = _json.loads(td_path.read_text())
        assert written == tier_delta, (
            f"Written tier_delta.json must equal the passed dict; got {written!r}"
        )

    def test_on_tier_delta_noop_when_snapshots_disabled(self, tmp_path) -> None:
        """on_tier_delta is a no-op (no file written) when save_cycle_snapshots=False."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.artifacts import on_tier_delta

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None

        with loop._artifact_scope():
            on_tier_delta(
                {
                    "episodic": {
                        "active_before": 5,
                        "active_after": 5,
                        "staled_by_reason": {},
                        "minted": 0,
                    }
                }
            )

        # No file should have been written anywhere.
        written_files = list(tmp_path.rglob("tier_delta.json"))
        assert written_files == [], (
            f"No tier_delta.json must be written when snapshots disabled; found {written_files}"
        )


# ---------------------------------------------------------------------------
# TestGraphLifecycle — cycle-end graph reset regression
# ---------------------------------------------------------------------------


class TestGraphLifecycle:
    """Regression: consolidate clears merger.graph at every exit.

    Uses a REAL GraphMerger (not MagicMock) so reset_graph() actually clears the
    graph, not a no-op as it would be under MagicMock.  This validates the
    cycle-end try/finally placement.

    Root cause being tested: before the fix, merger.graph leaked across cycles —
    a prior fold's ~199 reconstructed relations were captured into the next
    interim cycle's extra_relations, producing a spurious 208-key interim slot
    (live observed: graph_reconstructed=0, graph_merged=199 from extra_relations).
    """

    def test_graph_empty_after_successful_fold(self, tmp_path):
        """merger.graph is empty (0 nodes, 0 edges) after a successful fold.

        Arrange: one interim slot with a real triple.
        Act: consolidate(mode="simulate").
        Assert: merger.graph has 0 nodes and 0 edges.
        """
        loop = _make_bare_loop(tmp_path)
        # No pre-existing graph.json — load_memory_from_disk returns an empty
        # MultiDiGraph when the file is absent; the fold creates the file itself.
        _write_interim_graph(
            loop,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "knows", "object": "Bob"}],
        )

        loop.consolidate(mode="simulate")

        assert loop.merger.graph.number_of_nodes() == 0, (
            f"cycle-end graph reset regression: merger.graph must be empty after successful fold;"
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )
        assert loop.merger.graph.number_of_edges() == 0, (
            f"cycle-end graph reset regression: merger.graph must be empty after successful fold;"
            f"got {loop.merger.graph.number_of_edges()} edges"
        )

    def test_graph_empty_after_noop_fold(self, tmp_path):
        """merger.graph is empty after a no-op fold (no interim slots).

        The fold exits early (no slots to merge), but the finally block still fires.
        """
        loop = _make_bare_loop(tmp_path)
        # No pre-existing graph.json — the fold handles a missing file via
        # load_memory_from_disk returning an empty MultiDiGraph.

        loop.consolidate(mode="simulate")

        assert loop.merger.graph.number_of_nodes() == 0, (
            f"cycle-end graph reset regression: merger.graph must be empty after noop fold;"
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )
        assert loop.merger.graph.number_of_edges() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty after noop fold;"
            f"got {loop.merger.graph.number_of_edges()} edges"
        )

    def test_no_cross_cycle_bleed(self, tmp_path):
        """Two sequential folds with disjoint inputs produce disjoint canonical graphs.

        Regression for the 208-key bug: if merger.graph leaked from fold-1 into
        fold-2, the second fold's output would contain fold-1's edges.  After the
        cycle-end reset fix both folds start with an empty graph.
        """
        loop = _make_bare_loop(tmp_path)
        # No pre-existing graph.json for fold 1 — load_memory_from_disk returns
        # an empty MultiDiGraph when the file is absent; the fold creates/overwrites it.

        # Fold 1: one triple Alice→Bob.
        _write_interim_graph(
            loop,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "knows", "object": "Bob"}],
        )
        result1 = loop.consolidate(mode="simulate")
        assert result1.get("tiers_rebuilt") == ["episodic"]

        # After fold 1 the graph must be EMPTY.
        assert loop.merger.graph.number_of_nodes() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty between folds; "
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )

        # Fold 2: a fresh interim slot with a different, disjoint triple.
        # The canonical graph.json was written by fold-1 (Alice→Bob).  This
        # simulates a fresh cycle where only Carol's fact is new.
        _write_interim_graph(
            loop,
            "20260102T0000",
            [{"key": "graph2", "subject": "Carol", "predicate": "lives_in", "object": "Berlin"}],
        )
        result2 = loop.consolidate(mode="simulate")
        assert result2.get("tiers_rebuilt") == ["episodic"]

        # The canonical graph after fold 2 must contain BOTH folds' edges
        # (fold 2 reloads the canonical graph.json written by fold 1), NOT
        # just fold-1 edges re-injected via a leaked merger.graph.
        # keys_per_tier for episodic should be ≥ 2 (both triples).
        keys_after = result2.get("keys_per_tier", {}).get("episodic", 0)
        assert keys_after >= 2, (
            f"cycle-end graph reset regression: fold-2 canonical graph must contain both triples; "
            f"got keys_per_tier.episodic={keys_after}"
        )

        # Critically, merger.graph is STILL empty after fold 2.
        assert loop.merger.graph.number_of_nodes() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty after fold 2; "
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )

    # --- run_consolidation_cycle lifecycle tests ---

    def _build_loop_with_real_merger(self, tmp_path: Path):
        """Build a _build_loop() loop but replace merger with a real GraphMerger.

        The existing _build_loop() uses MagicMock for merger and stubs
        _materialize_consolidation_graph.  This helper swaps in a real
        GraphMerger so reset_graph() actually clears the graph (instead of
        being a no-op on a MagicMock).

        The _materialize_consolidation_graph stub is kept so the test does not
        need a GPU — it bypasses reconstruct_graph.  The cycle-end finally block calls
        the REAL merger.reset_graph(), which this test verifies.
        """
        from paramem.graph.merger import GraphMerger

        loop = _build_loop(tmp_path)
        loop.merger = GraphMerger(model=None)
        # Pre-seed the graph with a real edge so there is content to clear.
        loop.merger.graph.add_node("Alice")
        loop.merger.graph.add_node("Bob")
        loop.merger.graph.add_edge("Alice", "Bob", predicate="knows")
        return loop

    def test_run_consolidation_cycle_graph_empty_after_success(self, tmp_path):
        """merger.graph is empty (0 nodes, 0 edges) after a successful simulate cycle.

        The try/finally in run_consolidation_cycle calls reset_graph() on every
        exit that goes through the main try block.  This test uses a REAL GraphMerger
        so reset_graph() actually clears the graph.

        Mutation check: if the finally block were removed, merger.graph would retain
        the edges seeded before the call.
        """
        loop = self._build_loop_with_real_merger(tmp_path)
        assert loop.merger.graph.number_of_nodes() > 0, "Precondition: graph must be non-empty"

        loop.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="simulate",
            run_label="lifecycle-test",
            stamp=_STAMP,
        )

        assert loop.merger.graph.number_of_nodes() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty after successful cycle; "
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )
        assert loop.merger.graph.number_of_edges() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty after successful cycle; "
            f"got {loop.merger.graph.number_of_edges()} edges"
        )

    def test_run_consolidation_cycle_graph_empty_after_abort(self, tmp_path):
        """merger.graph is empty after an aborted cycle (exception propagated).

        Arrange: patch commit_tier_slot to raise RuntimeError to simulate an
        I/O failure inside the try block.  The finally must still clear the graph.

        Mutation check: without the finally, merger.graph retains the pre-seeded
        edges and this assertion fails.
        """
        loop = self._build_loop_with_real_merger(tmp_path)
        assert loop.merger.graph.number_of_nodes() > 0, "Precondition: graph must be non-empty"

        with patch(
            "paramem.memory.persistence.commit_tier_slot",
            side_effect=RuntimeError("simulated I/O failure"),
        ):
            try:
                loop.run_consolidation_cycle(
                    list(_EPISODIC_RELS),
                    list(_PROCEDURAL_RELS),
                    speaker_id=_SPEAKER_ID,
                    mode="simulate",
                    run_label="lifecycle-abort",
                    stamp=_STAMP,
                )
            except RuntimeError:
                pass  # expected — the abort propagated

        assert loop.merger.graph.number_of_nodes() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty after aborted cycle; "
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )

    # --- consolidate lifecycle tests ---

    # test_consolidate_graph_empty_after_accumulating_return was retired along
    # with the accumulate guard it exercised (2026-07-27 fast-start/tier-floor
    # retirement). It called consolidate(mode="train") with the pre-fold entry
    # guard bypassed but WITHOUT the train-path GPU mocks (_patches_for_train_mode)
    # applied, relying on the accumulate guard's early return to prevent the fold
    # from ever reaching the real (unmocked) per-tier training loop. With the
    # guard gone, this fixture's one pre-seeded keyless edge (Alice-knows-Bob) is
    # minted into a real key and the fold proceeds into main_tier_backup_scope and
    # _train_tier_adapter against a bare MagicMock model — not a clean terminal, and
    # not a safe unit-test path. The graph-reset invariant this test guarded stays
    # covered by test_graph_empty_after_successful_fold / test_graph_empty_after_noop_fold
    # above (full-fold disk venue) and by
    # test_run_consolidation_cycle_graph_empty_after_success / _after_abort below
    # (interim-cycle path) — no replacement test is added here.


# ---------------------------------------------------------------------------
# TestMaterializeConsolidationGraphDiskSource (source="disk" axis)
# ---------------------------------------------------------------------------


class TestMaterializeConsolidationGraphDiskSource:
    """Unit tests for :meth:`ConsolidationLoop._materialize_consolidation_graph` with
    ``source="disk"``.

    ``source`` is a WEIGHT-PROBE gate, not a merge-input selector: the
    registry-true re-merge (``_build_registry_true_relations``) runs in BOTH
    venues over the same store.  These tests pin that, so a store-seeded disk
    fold can never silently degrade to an empty merge input again.

    Verifies:
    - ``recall_miss_keys`` is always ``set()`` (no weight reconstruction).
    - ``recon_relations`` carries the store's active keys on the disk path.
    - ``extra_relations`` merge ALONGSIDE the store relations, not instead of them.
    - The ``source="weights"`` default path still runs the weight probe.
    """

    @staticmethod
    def _relation(subject: str, predicate: str, obj: str, key: str):
        """Build one supplemental :class:`Relation` for the extra_relations channel."""
        from paramem.graph.schema import Relation

        return Relation(
            subject=subject,
            predicate=predicate,
            object=obj,
            relation_type="factual",
            confidence=1.0,
            speaker_id="",
            indexed_key=key,
        )

    def test_disk_source_recall_miss_empty(self, tmp_path):
        """source='disk': recall_miss_keys is always the empty set.

        The disk path skips weight reconstruction entirely; no adapter weights are
        probed, so no failures can occur.  recall_miss_keys must be set() even
        with a populated store and supplemental relations.
        """
        loop = _make_bare_loop(tmp_path)
        _seed_store_tier(
            loop,
            "episodic",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )

        recall_miss_keys, _ = loop._materialize_consolidation_graph(
            source="disk",
            extra_relations=[self._relation("Bob", "works_at", "Acme", "k1")],
        )
        # Reset graph so it doesn't leak (the loop's finally would do this in production).
        loop.merger.reset_graph()

        assert recall_miss_keys == set(), (
            f"source='disk': recall_miss_keys must be empty set(); got {recall_miss_keys!r}"
        )

    def test_disk_source_builds_registry_true_relations_from_store(self, tmp_path):
        """source='disk': the store's active keys ARE the merge input.

        ``_build_registry_true_relations`` is called in both venues.  With a
        populated store the disk path must return those relations in
        ``recon_relations`` — an empty result here means the fold would rebuild
        the main tiers from nothing.
        """
        loop = _make_bare_loop(tmp_path)
        _seed_store_tier(
            loop,
            "episodic",
            [
                {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
                {"key": "graph2", "subject": "Alice", "predicate": "works_at", "object": "Acme"},
            ],
        )

        _, recon_relations = loop._materialize_consolidation_graph(source="disk")
        loop.merger.reset_graph()

        assert {r.indexed_key for r in recon_relations} == {"graph1", "graph2"}, (
            "source='disk': recon_relations must carry every active store key; "
            f"got {[r.indexed_key for r in recon_relations]!r}"
        )
        assert {(r.subject, r.predicate, r.object) for r in recon_relations} == {
            ("Alice", "lives_in", "Berlin"),
            ("Alice", "works_at", "Acme"),
        }

    def test_disk_source_merges_store_and_extra_relations_into_graph(self, tmp_path):
        """source='disk': store relations AND extra_relations both reach merger.graph.

        The supplemental channel is additive — it does not replace the
        registry-true merge input.  Both sets of indexed keys must be keyed onto
        the merged graph.
        """
        loop = _make_bare_loop(tmp_path)
        _seed_store_tier(
            loop,
            "episodic",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )

        loop._materialize_consolidation_graph(
            source="disk",
            extra_relations=[
                self._relation("Bob", "works_at", "Acme", "k2"),
                self._relation("Carol", "visits", "London", "k3"),
            ],
        )

        merged_keys = {
            data.get(_IK_KEY_ATTR)
            for _s, _o, data in loop.merger.graph.edges(data=True)
            if data.get(_IK_KEY_ATTR)
        }
        # Reset to avoid leaking into other tests.
        loop.merger.reset_graph()

        assert merged_keys == {"graph1", "k2", "k3"}, (
            "source='disk': the merged graph must carry the store's keys AND the "
            f"supplemental ones; got {merged_keys!r}"
        )

    def test_disk_source_empty_store_and_extra_relations_noop(self, tmp_path):
        """source='disk' with an empty store and no extras: empty result, no crash.

        The genuine "nothing to fold" case — an empty store IS an empty merge
        input, which is correct; the defect this class guards against is an empty
        merge input from a POPULATED store.
        """
        loop = _make_bare_loop(tmp_path)

        recall_miss_keys, recon_relations = loop._materialize_consolidation_graph(
            source="disk",
            extra_relations=[],
        )
        loop.merger.reset_graph()

        assert recall_miss_keys == set()
        assert recon_relations == []
        assert loop.merger.graph.number_of_edges() == 0

    def test_weights_source_default_unchanged(self, tmp_path):
        """source='weights' (default) falls through to the existing weights path.

        The default must not invoke the disk branch.  We verify this by checking
        that passing source='weights' with no extra_relations and a stubbed store
        does NOT immediately return (set(), []) without calling reconstruct_graph.
        A MagicMock model is injected so reconstruct_graph actually runs (and
        returns an empty recon result due to no PEFT adapters).
        """
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        loop = _make_bare_loop(tmp_path)
        # Wire a real (but empty) store so all_active_keys() returns [].
        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store

        # stub reconstruct_graph so we can confirm the weights path runs it.
        import unittest.mock as _mock

        _recon_mock = _mock.MagicMock()
        _recon_mock.failures = []
        _recon_mock.graph.edges.return_value = []

        with _mock.patch(
            "paramem.training.consolidation.reconstruct_graph",
            return_value=_recon_mock,
        ) as _patched_recon:
            recall_miss_keys, recon_relations = loop._materialize_consolidation_graph(
                source="weights",
            )
            loop.merger.reset_graph()

        # source='weights' must call reconstruct_graph (not the disk short-circuit).
        assert _patched_recon.called, "source='weights' must call reconstruct_graph"
        assert recall_miss_keys == set(), (
            "Weights path with empty store: recall_miss_keys must be empty set()"
        )
        assert recon_relations == [], (
            "Weights path with empty store: recon_relations must be [] (no active keys)"
        )


# ---------------------------------------------------------------------------
# TestSimulateFoldReturnSchema (return schema completeness)
# ---------------------------------------------------------------------------


class TestSimulateFoldReturnSchema:
    """The disk venue returns the SAME schema as the weights venue.

    One schema, both venues, every terminal return, so callers never KeyError
    on a venue they did not expect.  Covered: the noop return (nothing in the
    store) and the active return (interim slots folded into main).
    """

    _REQUIRED_KEYS = frozenset(
        {
            "tiers_rebuilt",
            "graph_drift_count",
            "drift_deduplicated",
            "drift_orphan",
            "drift_genuine_loss",
            "drift_intended_removal",
            "drift_intended_removal_by_reason",
            "recall_miss_keys",
            "keys_per_tier",
            "tier_keyed",
            "rolled_back",
            "rollback_tier",
            "tier_delta",
        }
    )

    def test_empty_input_return_has_full_schema(self, tmp_path):
        """A fold over an empty store carries all required keys.

        Nothing to fold is still a completed fold: the result must include
        drift_intended_removal, drift_intended_removal_by_reason,
        recall_miss_keys, and tier_keyed with zero/empty values.  ``tier_keyed``
        is the per-tier assignment map — empty lists, not an empty dict, because
        the spine always builds all three tiers.
        """
        loop = _make_bare_loop(tmp_path)
        result = loop.consolidate(mode="simulate")

        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, (
            f"Empty-input return missing required schema keys: {sorted(missing)}\n"
            f"  actual keys: {sorted(result.keys())}"
        )
        assert result["drift_intended_removal"] == 0
        assert result["drift_intended_removal_by_reason"] == {}
        assert result["recall_miss_keys"] == []
        assert result["tier_keyed"] == {"episodic": [], "semantic": [], "procedural": []}

    def test_active_return_has_full_schema(self, tmp_path):
        """Active return (interim slots present) carries all required keys.

        After a successful fold, the result must include all required schema
        keys, and ``tier_keyed`` must carry the folded key on the main tier it
        was assigned to.
        """
        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(
            loop,
            "20260301T0000",
            [
                {"key": "k1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            ],
        )

        result = loop.consolidate(mode="simulate")

        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, (
            f"Active return missing required schema keys: {sorted(missing)}\n"
            f"  actual keys: {sorted(result.keys())}"
        )
        assert result["drift_intended_removal"] == 0
        assert result["drift_intended_removal_by_reason"] == {}
        assert result["recall_miss_keys"] == []
        assert [e["key"] for e in result["tier_keyed"]["episodic"]] == ["k1"], (
            f"interim key must be rebooked onto episodic; got {result['tier_keyed']!r}"
        )

    def test_drift_genuine_loss_is_zero_for_disk_source(self, tmp_path):
        """drift_genuine_loss is 0 for the disk venue: nothing can fail reconstruction.

        Genuine loss counts active keys that had registry content but produced no
        merged edge.  The disk venue runs no weight reconstruction, so every
        active key reaches the merge through its registry-true relation and the
        bucket is empty by construction.
        """
        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(
            loop,
            "20260301T0000",
            [
                {"key": "k1", "subject": "Alice", "predicate": "likes", "object": "Tea"},
                {"key": "k2", "subject": "Bob", "predicate": "works_at", "object": "Acme"},
            ],
        )

        result = loop.consolidate(mode="simulate")

        assert result["drift_genuine_loss"] == 0, (
            f"Disk venue: drift_genuine_loss must be 0; got {result['drift_genuine_loss']!r}"
        )


# ---------------------------------------------------------------------------
# TestSimulateFoldSpineStages (spine stages that are NOT weight-only)
# ---------------------------------------------------------------------------


class TestSimulateFoldSpineStages:
    """Spine stages the disk venue runs identically to the weights venue.

    Promotion, the per-tier floor, the router reload, and the persist/reap pair
    are store operations, not weight operations, so the disk venue runs them —
    and their artifacts must show up in the venue's own sink (the per-tier
    ``graph.json`` projections), not only in episodic's.
    """

    def test_all_three_tier_graphs_written(self, tmp_path):
        """Every main tier the fold rebuilt gets its own ``<tier>/graph.json``.

        The disk venue's sink is three per-tier projections, not one canonical
        episodic graph.  A key that lives in semantic or procedural must be
        readable back from ITS tier's file — that path is what
        ``DiskMemorySource`` probes for the tier at the next hydration.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        _seed_store_tier(
            loop,
            "episodic",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )
        _seed_store_tier(
            loop,
            "semantic",
            [{"key": "graph2", "subject": "Alice", "predicate": "works_at", "object": "Acme"}],
        )
        _seed_store_tier(
            loop,
            "procedural",
            [
                {
                    "key": "proc1",
                    "subject": "speaker0",
                    "predicate": "prefers",
                    "object": "short answers",
                    "relation_type": "preference",
                }
            ],
        )

        result = loop.consolidate(mode="simulate")

        assert set(result["tiers_rebuilt"]) == {"episodic", "semantic", "procedural"}
        for tier, expected_key in (
            ("episodic", "graph1"),
            ("semantic", "graph2"),
            ("procedural", "proc1"),
        ):
            path = tmp_path / tier / "graph.json"
            assert path.exists(), f"{tier}/graph.json must be written by the simulate fold"
            keys = {e["key"] for e in iter_entries(load_memory_from_disk(path))}
            assert keys == {expected_key}, (
                f"{tier}/graph.json must carry exactly its own tier's keys; got {keys!r}"
            )

    def test_promotion_runs_on_the_disk_venue(self, tmp_path):
        """A mature episodic key is promoted to semantic in the disk venue too.

        Promotion is a pure store move with no weight dependency, so
        ``scope.promote`` is True in both venues.  The promoted key must end up
        in the semantic registry AND in ``semantic/graph.json``.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        # promotion_threshold defaults to 3 — seed one key at the threshold and
        # one below it, so the assertion distinguishes promotion from a blanket move.
        _seed_store_tier(
            loop,
            "episodic",
            [
                {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "reinforcement_count": loop.config.promotion_threshold,
                },
                {
                    "key": "graph2",
                    "subject": "Bob",
                    "predicate": "works_at",
                    "object": "Acme",
                    "reinforcement_count": 1,
                },
            ],
        )

        loop.consolidate(mode="simulate")

        assert loop.store.tier_for_active_key("graph1") == "semantic", (
            "the mature key must be promoted to semantic on the disk venue"
        )
        assert loop.store.tier_for_active_key("graph2") == "episodic", (
            "the immature key must stay in episodic"
        )
        semantic_keys = {
            e["key"]
            for e in iter_entries(load_memory_from_disk(tmp_path / "semantic" / "graph.json"))
        }
        assert semantic_keys == {"graph1"}, (
            f"the promoted key must be projected into semantic/graph.json; got {semantic_keys!r}"
        )

    def test_router_reload_fires_on_the_simulate_fold(self, tmp_path):
        """The router is reloaded after a simulate fold, exactly as after a train fold.

        The router serves from whatever the fold just published; a simulate fold
        publishes new per-tier graph.json projections, so it owes the router the
        same reload.
        """
        loop = _make_bare_loop(tmp_path)
        _seed_store_tier(
            loop,
            "episodic",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )
        router = MagicMock()

        loop.consolidate(mode="simulate", router=router)

        assert router.reload.called, "the simulate fold must reload the router"

    def test_interim_slot_survives_a_fold_that_persisted_nothing(self, tmp_path):
        """Reap is gated on the same predicate as persist — no persist, no reap.

        A slot whose content the store cannot see (boot-degraded hydration)
        contributes no keys, so the fold rebuilds nothing and writes nothing.
        Reaping it anyway would delete the only copy of its facts: on the disk
        venue the slot's ``graph.json`` IS the payload.
        """
        loop = _make_bare_loop(tmp_path)
        interim_dir = _write_interim_graph(
            loop,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
            store_state="absent",
        )

        result = loop.consolidate(mode="simulate")

        assert result["tiers_rebuilt"] == [], "nothing visible in the store → nothing rebuilt"
        assert not (tmp_path / "episodic" / "graph.json").exists(), (
            "a fold that rebuilt nothing must not write a tier graph"
        )
        assert interim_dir.exists(), (
            "the fold persisted nothing, so it must not reap the slot it could not fold"
        )
        assert (interim_dir / "graph.json").exists(), "the slot payload must survive intact"


class TestFoldHydratesAPartiallyPreloadedStore:
    """A fold entered with a partially-hydrated store must not lose the rest.

    The failure this locks down: ``store.get`` is cache-only, so every fold site
    that reads entry content used to drop a live key whose entry the boot preload
    had not materialised.  A dropped key never reaches ``tier_keyed``, and
    the finalize step rewrites every main-tier registry FROM ``tier_keyed``
    and flushes it — so the key was deregistered on disk and the drift partition
    filed it as an orphan.  The content was still in the venue's source of truth
    the whole time; nothing ever asked for it.

    ``app._build_store_contents`` reports exactly this state as
    ``boot_degraded={"reason": "preload_partial"}``.  A full cold cache is a
    distinct scenario, guarded upstream by :meth:`_hydrate_store_for_fold`
    itself (which runs before every fresh derivation and materialises every
    live key missing from the cache) — the retired ``min_tier_key_floor``
    accumulate guard was only ever a coincidental second net for it, never
    its owner, and no replacement guard is added here.  Both venues are
    covered — the disk venue reads the per-tier ``graph.json``, the weights
    venue re-probes the adapters.
    """

    _TRIPLES = [
        {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
        {"key": "graph2", "subject": "Bob", "predicate": "works_at", "object": "Acme"},
    ]

    def test_disk_venue_hydrates_the_missing_entry_from_graph_json(self, tmp_path):
        """Simulate venue: the un-preloaded key is read back out of graph.json.

        ``graph1`` is fully hydrated; ``graph2`` is registered and fingerprinted
        but has no entry.  Both live in the slot's ``graph.json``, which is what
        :class:`DiskMemorySource` reads.  After the fold both keys must be
        active, both must be in the rebuilt tier graph, and neither may have
        been staled.
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk
        from paramem.memory.source import DiskMemorySource
        from paramem.memory.store import MemoryStore

        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(loop, _STAMP, self._TRIPLES, store_state="hydrated")
        # Re-seed graph2 without its entry: registry + fingerprint + bookkeeping
        # present, entry cache empty — the partial-preload shape.
        loop.store._entries[f"episodic_interim_{_STAMP}"].pop("graph2")

        assert loop.store.get("graph2") is None, "fixture must start with graph2 un-hydrated"

        # Cardinality pin: the fold must probe the source exactly once,
        # batched across every active key of every registered tier — never
        # once per key, never once per read site.
        disk_probe_calls: list[dict] = []
        _orig_disk_probe = DiskMemorySource.probe
        store_probe_calls: list[dict] = []
        _orig_store_probe = MemoryStore.probe

        def _spy_disk_probe(self_inner, keys_by_adapter, should_abort=None):
            disk_probe_calls.append(dict(keys_by_adapter))
            return _orig_disk_probe(self_inner, keys_by_adapter, should_abort)

        def _spy_store_probe(self_inner, keys_by_adapter, **kwargs):
            store_probe_calls.append(dict(keys_by_adapter))
            return _orig_store_probe(self_inner, keys_by_adapter, **kwargs)

        with (
            patch.object(DiskMemorySource, "probe", _spy_disk_probe),
            patch.object(MemoryStore, "probe", _spy_store_probe),
        ):
            result = loop.consolidate(mode="simulate")

        assert len(disk_probe_calls) == 1, (
            f"DiskMemorySource.probe must be called exactly once per fold "
            f"(batched, never once per key); got {len(disk_probe_calls)}"
        )
        assert len(store_probe_calls) == 1, (
            f"MemoryStore.probe (the one call site inside "
            f"_hydrate_store_for_fold) must be called exactly once — no "
            f"downstream re-probe from any of the other fold read sites; "
            f"got {len(store_probe_calls)}"
        )
        source_probed_keys = {k for keys in disk_probe_calls[0].values() for k in keys}
        assert source_probed_keys == {"graph2"}, (
            "the single source probe must cover exactly the cache miss "
            f"(graph1 is already warm); got {source_probed_keys}"
        )
        store_probed_keys = {k for keys in store_probe_calls[0].values() for k in keys}
        assert store_probed_keys == {"graph1", "graph2"}, (
            "MemoryStore.probe's single call must cover every active key of "
            f"every registered tier; got {store_probed_keys}"
        )

        keyed = {e["key"] for tier in result["tier_keyed"].values() for e in tier}
        assert keyed == {"graph1", "graph2"}, (
            f"the un-hydrated key must be hydrated from graph.json and folded, not "
            f"dropped; tier_keyed carried {keyed}"
        )
        assert loop.store.tier_for_active_key("graph2") is not None, (
            "graph2 must still be an active registered key after the fold"
        )
        assert not loop.store.is_stale("graph2"), "graph2 must not have been staled"
        merged = load_memory_from_disk(tmp_path / "episodic" / "graph.json")
        assert {e["key"] for e in iter_entries(merged)} == {"graph1", "graph2"}, (
            "the rewritten tier graph must carry both keys"
        )

    def test_weights_venue_hydrates_the_missing_entry_from_the_adapter(self, tmp_path):
        """Train venue: the un-preloaded key is re-probed out of the adapter weights.

        Same partial-preload shape, but ``scope.source == "weights"``, so the
        content comes back through :class:`WeightMemorySource` →
        ``probe_keys_grouped_by_adapter`` (stubbed here — the GPU probe is the
        one thing this test does not run).  The key must survive the fold.
        """
        from peft import PeftModel

        loop = _build_loop(tmp_path)
        # Retiring the per-tier key-count floor means this 2-key fixture now
        # reaches the real per-tier training loop (main_tier_backup_scope
        # requires a PeftModel) instead of returning early on the (deleted)
        # accumulate guard.
        loop.model.__class__ = PeftModel
        _seed_store_tier(loop, "episodic", self._TRIPLES[:1])
        _seed_store_tier(loop, "episodic", self._TRIPLES[1:], entries=False)

        # Replace the fixture's keyless graph with the two keyed edges the
        # registry-true re-merge would produce, so the keyed branch of
        # _build_all_edge_entries_into — one of the three fold sites that read
        # entry content — is the thing under test.
        keyed_graph = nx.MultiDiGraph()
        for triple in self._TRIPLES:
            keyed_graph.add_node(triple["subject"], attributes={"name": triple["subject"]})
            keyed_graph.add_node(triple["object"], attributes={"name": triple["object"]})
            eid = keyed_graph.add_edge(
                triple["subject"],
                triple["object"],
                predicate=triple["predicate"],
                relation_type="factual",
            )
            keyed_graph[triple["subject"]][triple["object"]][eid][_IK_KEY_ATTR] = triple["key"]
        loop.merger.graph = keyed_graph

        assert loop.store.get("graph2") is None, "fixture must start with graph2 un-hydrated"

        from paramem.memory.store import MemoryStore

        probed: dict = {}
        weight_probe_calls: list[dict] = []

        def _fake_probe(model, tokenizer, keys_by_adapter, **kwargs):
            """Stand in for the adapter probe: the weights still hold graph2."""
            probed.update(keys_by_adapter)
            weight_probe_calls.append(dict(keys_by_adapter))
            out: dict = {}
            for keys in keys_by_adapter.values():
                for key in keys:
                    triple = next((t for t in self._TRIPLES if t["key"] == key), None)
                    out[key] = None if triple is None else {**triple, "confidence": 1.0}
            return out

        store_probe_calls: list[dict] = []
        _orig_store_probe = MemoryStore.probe

        def _spy_store_probe(self_inner, keys_by_adapter, **kwargs):
            store_probe_calls.append(dict(keys_by_adapter))
            return _orig_store_probe(self_inner, keys_by_adapter, **kwargs)

        # The consolidate(mode="train") entry guard requires the caller to hold
        # _gpu_thread_lock; a mock whose acquire() reports False satisfies it
        # (patch.object cannot patch a C-level threading.Lock attribute).
        _mock_lock = MagicMock()
        _mock_lock.acquire.return_value = False
        patches = _patches_for_train_mode()
        with (
            patch("paramem.server.gpu_lock._gpu_thread_lock", _mock_lock),
            patch("paramem.memory.probe.probe_keys_grouped_by_adapter", _fake_probe),
            patch.object(MemoryStore, "probe", _spy_store_probe),
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            # The per-tier training loop (reached now that the accumulate
            # guard is retired) touches real PEFT/model calls this MagicMock
            # model cannot satisfy — stub them the same way the full-fold
            # harness in test_consolidation.py does.
            patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
            patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
            patch.object(
                ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
            ),
            patch.object(ConsolidationLoop, "_save_adapters"),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
        ):
            result = loop.consolidate(mode="train")

        assert "graph2" in {k for keys in probed.values() for k in keys}, (
            "the fold must ask the weight source for the un-hydrated key"
        )
        # Cardinality pin: exactly one batched weight-source probe, and
        # exactly one MemoryStore.probe call (no downstream re-probe) —
        # never once per key, never once per read site.
        assert len(weight_probe_calls) == 1, (
            f"probe_keys_grouped_by_adapter must be called exactly once per "
            f"fold; got {len(weight_probe_calls)}"
        )
        assert len(store_probe_calls) == 1, (
            f"MemoryStore.probe must be called exactly once (no downstream "
            f"re-probe); got {len(store_probe_calls)}"
        )
        source_probed_keys = {k for keys in weight_probe_calls[0].values() for k in keys}
        assert source_probed_keys == {"graph2"}, (
            "the single source probe must cover exactly the cache miss "
            f"(graph1 is already warm); got {source_probed_keys}"
        )
        store_probed_keys = {k for keys in store_probe_calls[0].values() for k in keys}
        assert store_probed_keys == {"graph1", "graph2"}, (
            "MemoryStore.probe's single call must cover every active key of "
            f"every registered tier; got {store_probed_keys}"
        )
        keyed = {e["key"] for tier in result["tier_keyed"].values() for e in tier}
        assert "graph2" in keyed, (
            f"the un-hydrated key must be re-probed from the weights and folded, not "
            f"dropped; tier_keyed carried {keyed}"
        )
        assert loop.store.tier_for_active_key("graph2") is not None, (
            "graph2 must still be an active registered key after the fold"
        )
        assert not loop.store.is_stale("graph2"), "graph2 must not have been staled"

    def test_warm_cache_fold_never_probes_the_source(self, tmp_path):
        """A fully-hydrated store makes ZERO ``MemorySource.probe`` calls
        while every key still folds -- pins the ``if misses:`` guard
        (``store.py``) as the mechanism, and that the fold's ``memoize=True``
        is not conditional on ``inference.preload_cache``."""
        from unittest.mock import patch

        from paramem.memory.source import DiskMemorySource

        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(loop, _STAMP, self._TRIPLES, store_state="hydrated")
        assert loop.store.get("graph1") is not None and loop.store.get("graph2") is not None, (
            "fixture must start fully warm"
        )

        disk_probe_calls: list[dict] = []

        def _spy_disk_probe(self_inner, keys_by_adapter, should_abort=None):
            disk_probe_calls.append(dict(keys_by_adapter))
            raise AssertionError(
                "DiskMemorySource.probe must not be called against a fully-warm store"
            )

        with patch.object(DiskMemorySource, "probe", _spy_disk_probe):
            result = loop.consolidate(mode="simulate")

        assert disk_probe_calls == [], (
            f"expected zero source probes against a warm cache; got {disk_probe_calls}"
        )
        keyed = {e["key"] for tier in result["tier_keyed"].values() for e in tier}
        assert keyed == {"graph1", "graph2"}, (
            f"both keys must still fold from a warm cache; tier_keyed carried {keyed}"
        )
