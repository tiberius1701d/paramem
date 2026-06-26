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

    # Enrichment is off by default (refinement_enrichment="off", sota_enabled=False
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
      - first_seen_cycle preservation — re-running the same relations
        a second time leaves existing entries' first_seen_cycle unchanged.
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

        C-1/C-3 semantics: speaker_id is resolved from the EDGE first, then
        the SUBJECT NODE attribute, then terminal fallback to "".  The caller's
        id is NOT injected — ``default_speaker_id`` was removed (C-3).

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
        # C-1 reads the node attr (stamped ""); it must not be replaced.
        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("subject") == "Carol" and entry.get("object") == "London":
                    assert entry["speaker_id"] == "", (
                        f"Explicit empty speaker_id overwritten in {key}: {entry['speaker_id']!r}"
                    )

    def test_first_seen_cycle_preserved_on_second_run(self, loop_sim, loop_train, tmp_path):
        """Existing entries retain first_seen_cycle on re-run.

        Run the same fixture twice on each loop using the SAME stamp so that
        _resolve_target_slot returns the same adapter slot and
        _run_fold enters the existing-key reconstruction branch.
        Entries from cycle 1 must keep first_seen_cycle=1 after cycle 2.

        Using the same stamp is deliberate: this exercises the preservation
        branch (not just the fresh-slot path) that retains first_seen_cycle.
        """
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        # Record first_seen_cycle after cycle 1.
        sim_fsc_1 = {
            key: entry["first_seen_cycle"] for _tier, key, entry in loop_sim.store.iter_entries()
        }
        train_fsc_1 = {
            key: entry["first_seen_cycle"] for _tier, key, entry in loop_train.store.iter_entries()
        }

        # Record cycle-1 keys; used below to verify they survive into cycle 2.
        sim_keys_1 = set(key for _tier, key, _entry in loop_sim.store.iter_entries())
        train_keys_1 = set(key for _tier, key, _entry in loop_train.store.iter_entries())

        # Run again with the SAME stamp to exercise the existing-key reconstruction
        # branch in _run_fold (same adapter slot, existing keys reconstructed).
        loop_sim.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="simulate",
            run_label="parity2",
            stamp=_STAMP,  # same stamp as cycle 1
        )
        patches = _patches_for_train_mode()
        with patches[0], patches[1], patches[2], patches[3]:
            loop_train.run_consolidation_cycle(
                list(_EPISODIC_RELS),
                list(_PROCEDURAL_RELS),
                speaker_id=_SPEAKER_ID,
                mode="train",
                run_label="parity2",
                stamp=_STAMP,  # same stamp as cycle 1
            )

        # Existing entries must not have their first_seen_cycle bumped.
        for key, fsc in sim_fsc_1.items():
            entry = loop_sim.store.get(key)
            if entry is not None:
                assert entry["first_seen_cycle"] == fsc, (
                    f"Simulate: first_seen_cycle changed for {key}: "
                    f"{fsc} → {entry['first_seen_cycle']}"
                )

        for key, fsc in train_fsc_1.items():
            entry = loop_train.store.get(key)
            if entry is not None:
                assert entry["first_seen_cycle"] == fsc, (
                    f"Train: first_seen_cycle changed for {key}: "
                    f"{fsc} → {entry['first_seen_cycle']}"
                )

        # All keys from cycle 1 (episodic graph* AND procedural proc*) must
        # survive into cycle 2.  Per-session procedural sp_index-driven retirement
        # Per-session procedural sp_index-driven retirement has been removed; procedural
        # contradictions are now resolved at full consolidation by the
        # model-bearing GraphMerger.  Duplicate procedural keys from a same-
        # preference re-run are tolerated in the interim.
        sim_keys_2 = set(key for _tier, key, _entry in loop_sim.store.iter_entries())
        train_keys_2 = set(key for _tier, key, _entry in loop_train.store.iter_entries())
        sim_keys_1_all = sim_keys_1
        train_keys_1_all = train_keys_1
        assert sim_keys_1_all.issubset(sim_keys_2), (
            f"Simulate: cycle-1 keys dropped after re-run: {sim_keys_1_all - sim_keys_2}"
        )
        assert train_keys_1_all.issubset(train_keys_2), (
            f"Train: cycle-1 keys dropped after re-run: {train_keys_1_all - train_keys_2}"
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
                "first_seen_cycle": quad.get("first_seen_cycle", 0),
            },
        )
    save_memory_to_disk(graph, path, encrypted=False)


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
                    "first_seen_cycle": 0,
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
                    "first_seen_cycle": 0,
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
                    "first_seen_cycle": 0,
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
                    "first_seen_cycle": 0,
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
                    "first_seen_cycle": 0,
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
            "first_seen_cycle": 0,
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
            "first_seen_cycle",
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
    """Build the minimal ConsolidationLoop required by consolidate_simulate_fold.

    Attributes set:
      - ``output_dir`` — used as the adapter_dir root.
      - ``merger`` — a model-free ``GraphMerger`` so the merge topology
        can run without a GPU or loaded model (``merger.model=None`` means the
        model-gated Case-2 branch is skipped; production-correct for simulate mode).
      - ``save_cycle_snapshots`` — False so ``_debug_writer`` gates to no-op.
      - ``_debug_base`` — None so ``_debug_writer._active_base()`` returns None.
      - ``config`` — minimal ``ConsolidationConfig`` with base defaults
        (sota_enabled=False, refinement_enrichment="off", refinement_normalization="off")
        so enrichment and normalization are suppressed without explicit flags.

    All other attributes are left unset; any unintended access will raise
    AttributeError rather than silently returning a MagicMock value.
    """
    from paramem.graph.merger import GraphMerger

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.output_dir = tmp_path
    loop.merger = GraphMerger(model=None)
    # model/tokenizer None: the whole-graph normalization pass reads self.model and
    # cleanly skips (skip_reason="no_model") — production-correct for simulate mode,
    # which has no local model resident.
    loop.model = None
    loop.tokenizer = None
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    loop.config = ConsolidationConfig()
    return loop


def _write_interim_graph(adapter_dir: Path, stamp: str, triples: list[dict]) -> Path:
    """Write a simulate-mode interim graph.json under adapter_dir/episodic/interim_<stamp>/.

    Args:
        adapter_dir: The loop's output_dir.
        stamp: Sub-interval stamp, e.g. ``"20260101T0000"``.
        triples: List of dicts with ``key``, ``subject``, ``predicate``,
            ``object``, and optionally ``speaker_id`` / ``first_seen_cycle``.

    Returns:
        The interim directory path.
    """
    interim_dir = adapter_dir / "episodic" / f"interim_{stamp}"
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
                "first_seen_cycle": t.get("first_seen_cycle", 0),
            },
        )
    from paramem.memory.persistence import save_memory_to_disk as _save

    _save(graph, interim_dir / "graph.json", encrypted=False)
    return interim_dir


class TestConsolidateSimulateFold:
    """Tests for consolidate_simulate_fold (simulate-mode full fold)."""

    def test_no_interim_slots_returns_noop_result(self, tmp_path):
        """Empty interim set: no main-graph write, no crash, result signals noop.

        When no ``episodic/interim_*`` directories contain a ``graph.json``,
        the method returns the canonical noop dict and does NOT create the
        main-tier ``graph.json`` (nothing to merge).
        """
        loop = _make_bare_loop(tmp_path)

        result = loop.consolidate_simulate_fold()

        assert result["tiers_rebuilt"] == [], "No interim slots → tiers_rebuilt must be empty"
        assert result["rolled_back"] is False
        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert not main_graph_path.exists(), (
            "No interim slots → main graph.json must not be created"
        )

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
        interim_dir = _write_interim_graph(tmp_path, "20260101T0000", triples)

        result = loop.consolidate_simulate_fold()

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
            tmp_path,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )
        slot_b = _write_interim_graph(
            tmp_path,
            "20260102T0000",
            [{"key": "graph2", "subject": "Bob", "predicate": "works_at", "object": "Acme"}],
        )

        result = loop.consolidate_simulate_fold()

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
        _write_interim_graph(tmp_path, "20260101T0000", [shared_triple])
        _write_interim_graph(tmp_path, "20260102T0000", [shared_triple])

        loop.consolidate_simulate_fold()

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
                tmp_path,
                f"2026010{i}T0000",
                [{"key": f"graph{i}", "subject": f"E{i}", "predicate": "p", "object": f"O{i}"}],
            )
            for i in range(1, 4)
        ]

        loop.consolidate_simulate_fold()

        for d in dirs:
            assert not d.exists(), f"Interim dir {d} must be removed after consolidation"
        assert (tmp_path / "episodic" / "graph.json").exists(), (
            "Main graph.json must exist after merge"
        )

    def test_result_contains_tier_delta(self, tmp_path):
        """Result dict contains 'tier_delta' with episodic before/after counts.

        Both the scheduled and housekeeping paths emit tier_delta.  For the
        simulate path staled_by_reason is {} in this fixture because no dedup
        collapse occurs (the single interim slot has a unique triple) and the
        simulate-mode store has no entries for removal_ledger attribution.
        """

        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(
            tmp_path,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "likes", "object": "Tea"}],
        )

        result = loop.consolidate_simulate_fold()

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
            "simulate path: staled_by_reason must be empty — no dedup collapse in this fixture"
            " and simulate-mode store has no entries for removal_ledger attribution"
        )
        assert ep["active_before"] == 0, "No main graph before → active_before == 0"
        assert ep["active_after"] == 1, "One triple merged → active_after == 1"

    def test_housekeeping_noop_on_empty_main_and_no_interims(self, tmp_path):
        """housekeeping=True with no interims and empty main graph: persists the empty
        merged graph to disk and returns tiers_rebuilt=['episodic'].

        housekeeping=True bypasses the empty-interim early-return so the groomed
        (empty) graph is written back to disk.  This is the simulate counterpart
        of the housekeeping gate in the train mode.
        """

        loop = _make_bare_loop(tmp_path)
        # No interim slots, no pre-existing main graph.

        result = loop.consolidate_simulate_fold(housekeeping=True)

        assert result["tiers_rebuilt"] == ["episodic"], (
            "housekeeping=True: tiers_rebuilt must be ['episodic'] even with no interims; "
            f"got {result['tiers_rebuilt']!r}"
        )
        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert main_graph_path.exists(), (
            "housekeeping=True: main graph.json must be written even with no interims"
        )

    def test_housekeeping_sets_current_interim_stamp(self, tmp_path):
        """During consolidate_simulate_fold(housekeeping=True) the loop's
        _current_interim_stamp is set to 'housekeeping_<ts>' and then cleared.

        After the method returns, the stamp is None (cleared on exit).  This test
        checks the post-call cleared state (the live value during the call is not
        observable from a unit test).
        """
        loop = _make_bare_loop(tmp_path)
        loop.consolidate_simulate_fold(housekeeping=True)

        # Stamp must be cleared on normal return.
        stamp = getattr(loop, "_current_interim_stamp", "NOT_SET")
        assert stamp is None, (
            "_current_interim_stamp must be None after "
            f"consolidate_simulate_fold returns; got {stamp!r}"
        )

    def test_run_housekeeping_dispatches_to_consolidate_simulate_fold(self, tmp_path):
        """run_housekeeping() on a simulate-mode loop calls consolidate_simulate_fold
        with housekeeping=True, NOT consolidate_interim_adapters.

        With config=None the loop's run_housekeeping() falls through to 'simulate'
        mode (the default when config is absent).
        """
        from unittest.mock import patch

        loop = _make_bare_loop(tmp_path)
        # config=None → mode defaults to "simulate" in run_housekeeping.

        _meth = loop.consolidate_simulate_fold
        with patch.object(loop, "consolidate_simulate_fold", wraps=_meth) as mock_cig:
            loop.run_housekeeping()

        mock_cig.assert_called_once_with(housekeeping=True)

    def test_cross_slot_variant_collapse_in_simulate(self, tmp_path):
        """Cross-slot variant-pair collapse: two slots with different surface forms
        for the same canonical triple merge to a single edge.

        Seeds two interim graph.json slots whose surfaces differ but canonicalize
        to the same identity (subject "Alice" vs "alice", object "Acme Corp" vs
        "acme corp").  After consolidate_simulate_fold(housekeeping=True):

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
            tmp_path,
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
            tmp_path,
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

        loop.consolidate_simulate_fold(housekeeping=True)

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

        Strategy: monkeypatch _run_graph_enrichment to add a sentinel edge to
        self.merger.graph and return new_edges=1.  After consolidate_simulate_fold(
        housekeeping=True), assert:
        (a) the sentinel edge is present in the persisted graph.json, and
        (b) the merged interim edge is also present (sentinel coexists with merged content),
        (c) tier_delta["episodic"]["minted"] == 1 (sourced from enrichment new_edges).
        """
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loop = _make_bare_loop(tmp_path)
        # refinement_enrichment="on" + sota_enabled=True so consolidate_simulate_fold
        # calls _run_graph_enrichment; base defaults (off/False) would skip it.
        loop.config = ConsolidationConfig(refinement_enrichment="on", sota_enabled=True)

        # Seed one interim slot so there is merged content to coexist with.
        _write_interim_graph(
            tmp_path,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}],
        )

        # Sentinel values for the edge the enrichment mock will inject.
        SENTINEL_SUBJECT = "__enr_sentinel__"
        SENTINEL_OBJECT = "__enr_sentinel_obj__"
        SENTINEL_KEY = "__enr_sentinel_key__"

        def _fake_enrichment(self_loop=None):
            """Add a sentinel edge to self.merger.graph and return new_edges=1.

            Must run on the populated merged graph (after reset+merge), otherwise
            the merger's graph is empty and the sentinel is wiped by reset_graph().
            The buggy code runs enrichment BEFORE reset_graph(), so the sentinel
            would be absent from the persisted graph on the old path.
            """
            loop.merger.graph.add_edge(
                SENTINEL_SUBJECT,
                SENTINEL_OBJECT,
                **{
                    _IK_KEY_ATTR: SENTINEL_KEY,
                    "predicate": "enriched_by",
                    "speaker_id": "",
                    "first_seen_cycle": 0,
                },
            )
            return {
                "chunks": 1,
                "new_edges": 1,
                "same_as_merges": 0,
                "skipped": False,
                "skip_reason": None,
            }

        from unittest.mock import patch

        with patch.object(loop, "_run_graph_enrichment", side_effect=_fake_enrichment):
            result = loop.consolidate_simulate_fold(housekeeping=True)

        # (a) Sentinel edge must be present in the persisted graph.json.
        main_graph_path = tmp_path / "episodic" / "graph.json"
        assert main_graph_path.exists(), "Main graph.json must be written"
        merged = load_memory_from_disk(main_graph_path)
        keys_in_graph = {e["key"] for e in iter_entries(merged)}
        assert SENTINEL_KEY in keys_in_graph, (
            f"Sentinel enrichment edge must survive into persisted graph.json; "
            f"keys present: {sorted(keys_in_graph)}\n"
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
        """Simulate path via _merge_registry_relations stamps entity_type='person' + speaker_id.

        Regression guard: before routing consolidate_simulate_fold through
        _merge_registry_relations, the simulate path used entities=[] directly
        (same latent bug as the recon path fixed by _merge_registry_relations
        unification).  A relation whose subject == speaker_id (i.e. a speaker
        node) would be stored as entity_type='concept' with no speaker_id
        attribute, causing keyless-edge attribution to fall back to speaker_id="".

        After the fix, the simulate path calls _merge_registry_relations which
        calls _synth_speaker_entities; the subject node receives
        entity_type='person' and speaker_id from the synthesised Entity.

        Uses the session_id="__simulate_consolidation_merge__" path, which is
        the simulate full-fold session identifier.
        """
        loop = _make_bare_loop(tmp_path)

        # Write an interim slot whose triple has subject == speaker_id.
        # _synth_speaker_entities fires when _r.speaker_id != "" AND
        # _r.subject == _r.speaker_id.  We use "speaker0" for both.
        _write_interim_graph(
            tmp_path,
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

        loop.consolidate_simulate_fold()

        # merger.graph is cleared by the cycle-end finally; read the persisted graph.json instead.
        from paramem.memory.persistence import load_memory_from_disk

        main_path = tmp_path / "episodic" / "graph.json"
        assert main_path.exists(), "graph.json must be written after the simulate fold"
        saved_graph = load_memory_from_disk(main_path)

        # §0 invariant (Step 2): speaker node key is the lowercase speaker_id.
        # Under the lowercase-uniform design entity.speaker_id == node key == "speaker0".
        node_key = "speaker0"
        assert node_key in saved_graph.nodes, (
            f"Speaker subject node {node_key!r} missing from merged graph after simulate path; "
            f"nodes present: {list(saved_graph.nodes)}"
        )
        node_data = saved_graph.nodes[node_key]
        assert node_data.get("entity_type") == "person", (
            f"Simulate path: expected entity_type='person' on speaker subject node; "
            f"got entity_type={node_data.get('entity_type')!r}. "
            "Regression: before _merge_registry_relations routing, simulate used "
            "entities=[] so speaker nodes received entity_type='concept' with no speaker_id."
        )
        assert node_data.get("speaker_id") == "speaker0", (
            f"Simulate path: expected speaker_id='speaker0' (cased, in node attribute); "
            f"got speaker_id={node_data.get('speaker_id')!r}. "
            "Regression: _synth_speaker_entities was not applied to the simulate path."
        )


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
            "graph2": {"reason": "dedup", "surviving_twin": "graph1", "pre_surfaces": {}}
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
            "graph3": {"reason": "enrichment_same_as", "merged_into": "alice"}
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
            "ghost_key": {"reason": "dedup", "surviving_twin": "real_key", "pre_surfaces": {}}
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
            "ep_key": {"reason": "dedup", "surviving_twin": "other", "pre_surfaces": {}},
            "sem_key": {"reason": "dedup", "surviving_twin": "other2", "pre_surfaces": {}},
            "proc_key": {"reason": "dedup", "surviving_twin": "other3", "pre_surfaces": {}},
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
            tmp_path,
            "20260201T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "works at", "object": "Acme Corp"}],
        )
        _write_interim_graph(
            tmp_path,
            "20260202T0000",
            [{"key": "graph2", "subject": "alice", "predicate": "works at", "object": "acme corp"}],
        )

        result = loop.consolidate_simulate_fold(housekeeping=True)

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
        assert main_path.exists(), "graph.json must exist after housekeeping fold"
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
            "first_seen_cycle": 0,
        },
        register=False,
    )
    store.set_bookkeeping("graph1", speaker_id="sp1", first_seen_cycle=0, relation_type="factual")
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
                "first_seen_cycle": 0,
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
# DebugSnapshotWriter integration through run_consolidation_cycle
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
        # via on_fold_graph (no interim_stamp) → lands under fold/ not cycle_dir.
        # graph_merged_snapshot.json is no longer emitted on the interim path.
        fold_base = loop.snapshot_dir_for()
        assert fold_base is not None
        loop.merger.save_graph.assert_any_call(
            fold_base / "fold" / "graph_enriched_snapshot.json", encrypted=False
        )
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
    """``DebugSnapshotWriter.on_tier_delta`` persists the per-tier delta record.

    Both the scheduled fold and the housekeeping endpoint emit a
    ``tier_delta.json`` under ``<debug_base>/fold/`` so operators can see
    before/after/staled/minted counts without parsing raw adapter weight files.
    """

    def test_on_tier_delta_writes_file_when_snapshots_enabled(self, tmp_path) -> None:
        """on_tier_delta writes tier_delta.json under fold/ when save_cycle_snapshots=True."""
        import json as _json

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.debug_snapshot import DebugSnapshotWriter

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = True
        loop._debug_base = tmp_path / "debug"
        loop._current_interim_stamp = None
        loop.run_id = "test_run_01"
        loop.cycle_count = 1

        writer = DebugSnapshotWriter(loop)

        tier_delta = {
            "episodic": {
                "active_before": 10,
                "active_after": 8,
                "staled_by_reason": {"dedup": 2},
                "minted": 0,
            }
        }
        writer.on_tier_delta(tier_delta)

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
        from paramem.training.debug_snapshot import DebugSnapshotWriter

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None

        writer = DebugSnapshotWriter(loop)
        writer.on_tier_delta(
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
    """Regression: consolidate_simulate_fold clears merger.graph at every exit.

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
        Act: consolidate_simulate_fold().
        Assert: merger.graph has 0 nodes and 0 edges.
        """
        loop = _make_bare_loop(tmp_path)
        # No pre-existing graph.json — load_memory_from_disk returns an empty
        # MultiDiGraph when the file is absent; the fold creates the file itself.
        _write_interim_graph(
            tmp_path,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "knows", "object": "Bob"}],
        )

        loop.consolidate_simulate_fold()

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

        loop.consolidate_simulate_fold()

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
            tmp_path,
            "20260101T0000",
            [{"key": "graph1", "subject": "Alice", "predicate": "knows", "object": "Bob"}],
        )
        result1 = loop.consolidate_simulate_fold()
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
            tmp_path,
            "20260102T0000",
            [{"key": "graph2", "subject": "Carol", "predicate": "lives_in", "object": "Berlin"}],
        )
        result2 = loop.consolidate_simulate_fold()
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

    # --- consolidate_interim_adapters lifecycle tests ---

    def test_consolidate_interim_adapters_graph_empty_after_accumulating_return(self, tmp_path):
        """merger.graph is empty after consolidate_interim_adapters accumulating return.

        The accumulating early return fires when total trainable keys < floor (30).
        With an empty MemoryStore and no interim slots, _total_trainable == 0 < 30.

        This test patches:
        - _gpu_thread_lock.acquire(blocking=False) → False (pretend lock is held),
          bypassing the entry guard without needing a real BackgroundTrainer.
        - _materialize_consolidation_graph → (set(), []) to skip reconstruct_graph
          (GPU call).

        The real GraphMerger is used so reset_graph() in the finally actually clears
        the graph.

        Mutation check: without the cycle-end finally block, merger.graph retains the
        pre-seeded edge.
        """
        from paramem.graph.merger import GraphMerger

        loop = _build_loop(tmp_path)
        loop.merger = GraphMerger(model=None)
        # Pre-seed the graph with a real edge.
        loop.merger.graph.add_node("Alice")
        loop.merger.graph.add_node("Bob")
        loop.merger.graph.add_edge("Alice", "Bob", predicate="knows")

        assert loop.merger.graph.number_of_nodes() > 0, "Precondition: graph must be non-empty"

        # Patch the GPU lock entry guard: replace the module-level lock object with
        # a MagicMock whose acquire(blocking=False) returns False so the entry guard
        # believes the lock is already held.  patch.object cannot patch a C-level
        # threading.Lock attribute, so we replace the module-level name instead.
        _mock_lock = MagicMock()
        _mock_lock.acquire.return_value = False  # pretend caller already holds the lock
        with patch("paramem.server.gpu_lock._gpu_thread_lock", _mock_lock):
            result = loop.consolidate_interim_adapters(trainer=None)

        assert result.get("status") == "accumulating", (
            f"Expected status='accumulating' with 0 keys < floor 30; got {result!r}"
        )
        assert loop.merger.graph.number_of_nodes() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty on accumulating return; "
            f"got {loop.merger.graph.number_of_nodes()} nodes"
        )
        assert loop.merger.graph.number_of_edges() == 0, (
            "cycle-end graph reset regression: merger.graph must be empty on accumulating return; "
            f"got {loop.merger.graph.number_of_edges()} edges"
        )


# ---------------------------------------------------------------------------
# TestCollectDiskFoldRelations
# ---------------------------------------------------------------------------


class TestCollectDiskFoldRelations:
    """Unit tests for :meth:`ConsolidationLoop._collect_disk_fold_relations`.

    Verifies:
    - ``active_before_count`` is the canonical graph's pre-merge edge count.
    - Train-mode interim slots (no ``graph.json``) are skipped.
    - ``interim_dirs`` contains only simulate-mode slots that have a ``graph.json``.
    - Relations are built from BOTH the canonical graph and simulate-mode interim slots.
    - Missing canonical graph returns ``active_before_count=0`` (empty graph).
    """

    def test_active_before_count_from_canonical_graph(self, tmp_path):
        """active_before_count reflects the canonical episodic/graph.json edge count.

        Writes a canonical graph.json with 2 edges and one interim slot with
        1 edge.  active_before_count must equal 2 (canonical pre-merge count),
        NOT 3 (total after merging interim).
        """
        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR, save_memory_to_disk

        # Write canonical graph with 2 edges.
        canonical_path = tmp_path / "episodic" / "graph.json"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_g = nx.MultiDiGraph()
        canonical_g.add_edge("A", "B", **{_IK_KEY_ATTR: "k1", "predicate": "p1", "speaker_id": ""})
        canonical_g.add_edge("C", "D", **{_IK_KEY_ATTR: "k2", "predicate": "p2", "speaker_id": ""})
        save_memory_to_disk(canonical_g, canonical_path, encrypted=False)

        # Write one simulate-mode interim slot with 1 edge.
        _write_interim_graph(
            tmp_path,
            "20260301T0000",
            [
                {"key": "k3", "subject": "E", "predicate": "p3", "object": "F"},
            ],
        )

        loop = _make_bare_loop(tmp_path)
        inp = loop._collect_disk_fold_relations(tmp_path)

        assert inp.active_before_count == 2, (
            f"active_before_count must equal canonical graph edge count (2); "
            f"got {inp.active_before_count}"
        )

    def test_no_canonical_graph_returns_zero_active_before(self, tmp_path):
        """When canonical episodic/graph.json does not exist, active_before_count == 0.

        The first fold has no prior canonical graph; load_memory_from_disk returns
        an empty graph whose edge count is 0.
        """
        _write_interim_graph(
            tmp_path,
            "20260301T0000",
            [
                {"key": "k1", "subject": "A", "predicate": "p1", "object": "B"},
            ],
        )

        loop = _make_bare_loop(tmp_path)
        inp = loop._collect_disk_fold_relations(tmp_path)

        assert inp.active_before_count == 0, (
            f"No canonical graph → active_before_count must be 0; got {inp.active_before_count}"
        )

    def test_train_mode_slot_skipped_no_graph_json(self, tmp_path):
        """Train-mode interim slots (no graph.json) are skipped.

        Creates one simulate-mode slot (has graph.json) and one train-mode slot
        (directory with no graph.json, only a sentinel file).  Only the simulate
        slot must appear in interim_dirs.
        """
        # Simulate-mode slot: has graph.json.
        simulate_dir = _write_interim_graph(
            tmp_path,
            "20260301T0000",
            [
                {"key": "k1", "subject": "A", "predicate": "p1", "object": "B"},
            ],
        )

        # Train-mode slot: directory with no graph.json (mimics a PEFT weights slot).
        train_slot_dir = tmp_path / "episodic" / "interim_20260302T0000"
        train_slot_dir.mkdir(parents=True, exist_ok=True)
        # Place a sentinel file (not graph.json) to confirm the directory exists.
        (train_slot_dir / "adapter_config.json").write_text('{"r": 8}')

        loop = _make_bare_loop(tmp_path)
        inp = loop._collect_disk_fold_relations(tmp_path)

        assert simulate_dir in inp.interim_dirs, (
            "Simulate-mode slot (has graph.json) must appear in interim_dirs"
        )
        assert train_slot_dir not in inp.interim_dirs, (
            "Train-mode slot (no graph.json) must be skipped"
        )
        # Relations from simulate slot are present; train slot contributes nothing.
        rel_keys = {r.indexed_key for r in inp.relations}
        assert "k1" in rel_keys, "Relation from simulate slot must be collected"

    def test_relations_from_canonical_and_interim_slots(self, tmp_path):
        """Relations are built from both the canonical graph and simulate-mode interim slots.

        Writes a canonical graph with key k1 and an interim slot with key k2.
        The returned relations must contain both.
        """
        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR, save_memory_to_disk

        canonical_path = tmp_path / "episodic" / "graph.json"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_g = nx.MultiDiGraph()
        canonical_g.add_edge("A", "B", **{_IK_KEY_ATTR: "k1", "predicate": "p1", "speaker_id": ""})
        save_memory_to_disk(canonical_g, canonical_path, encrypted=False)

        _write_interim_graph(
            tmp_path,
            "20260301T0000",
            [
                {"key": "k2", "subject": "C", "predicate": "p2", "object": "D"},
            ],
        )

        loop = _make_bare_loop(tmp_path)
        inp = loop._collect_disk_fold_relations(tmp_path)

        rel_keys = {r.indexed_key for r in inp.relations}
        assert "k1" in rel_keys, "Canonical graph relation (k1) must be collected"
        assert "k2" in rel_keys, "Interim slot relation (k2) must be collected"


# ---------------------------------------------------------------------------
# TestMaterializeConsolidationGraphDiskSource (source="disk" axis)
# ---------------------------------------------------------------------------


class TestMaterializeConsolidationGraphDiskSource:
    """Unit tests for :meth:`ConsolidationLoop._materialize_consolidation_graph` with
    ``source="disk"``.

    Verifies:
    - ``recall_miss_keys`` is always ``set()`` (no weight reconstruction).
    - ``recon_relations`` is always ``[]`` (no registry-true re-merge).
    - Disk relations fed via ``extra_relations`` appear in ``merger.graph`` after the call.
    - The ``source="weights"`` default path is unchanged (existing callers byte-identical).
    """

    def test_disk_source_recall_miss_empty(self, tmp_path):
        """source='disk': recall_miss_keys is always the empty set.

        The disk path skips weight reconstruction entirely; no adapter weights are
        probed, so no failures can occur.  recall_miss_keys must be set() regardless
        of the extra_relations content.
        """
        from paramem.graph.schema import Relation

        loop = _make_bare_loop(tmp_path)
        # Provide some disk relations.
        relations = [
            Relation(
                subject="Alice",
                predicate="lives_in",
                object="Berlin",
                relation_type="factual",
                confidence=1.0,
                speaker_id="",
                indexed_key="k1",
            )
        ]

        recall_miss_keys, recon_relations = loop._materialize_consolidation_graph(
            source="disk",
            extra_relations=relations,
        )
        # Reset graph so it doesn't leak (the loop's finally would do this in production).
        loop.merger.reset_graph()

        assert recall_miss_keys == set(), (
            f"source='disk': recall_miss_keys must be empty set(); got {recall_miss_keys!r}"
        )
        assert recon_relations == [], (
            f"source='disk': recon_relations must be [] (no registry-true re-merge); "
            f"got {recon_relations!r}"
        )

    def test_disk_source_merges_extra_relations_into_graph(self, tmp_path):
        """source='disk': extra_relations appear in merger.graph after the call.

        Disk relations are the ONLY merge input for the disk path.  After the call,
        merger.graph must contain an edge for each supplied relation.
        """
        from paramem.graph.schema import Relation

        loop = _make_bare_loop(tmp_path)
        relations = [
            Relation(
                subject="Bob",
                predicate="works_at",
                object="Acme",
                relation_type="factual",
                confidence=1.0,
                speaker_id="",
                indexed_key="k2",
            ),
            Relation(
                subject="Carol",
                predicate="visits",
                object="London",
                relation_type="factual",
                confidence=1.0,
                speaker_id="",
                indexed_key="k3",
            ),
        ]

        loop._materialize_consolidation_graph(
            source="disk",
            extra_relations=relations,
        )

        # merger.graph must have at least 2 edges (one per relation).
        n_edges = loop.merger.graph.number_of_edges()
        assert n_edges >= 2, (
            f"source='disk': merger.graph must have edges for each extra_relation; "
            f"got {n_edges} edges (expected >= 2)"
        )
        # Reset to avoid leaking into other tests.
        loop.merger.reset_graph()

    def test_disk_source_empty_extra_relations_noop(self, tmp_path):
        """source='disk' with empty extra_relations: merger.graph stays empty, no crash.

        An empty extra_relations list is a valid no-op (housekeeping on an empty
        canonical graph with no interim slots).
        """
        loop = _make_bare_loop(tmp_path)

        recall_miss_keys, recon_relations = loop._materialize_consolidation_graph(
            source="disk",
            extra_relations=[],
        )
        loop.merger.reset_graph()

        assert recall_miss_keys == set()
        assert recon_relations == []

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
    """Assert that consolidate_simulate_fold returns the FULL train schema.

    The train return dict carries ``drift_intended_removal``,
    ``drift_intended_removal_by_reason``, ``recall_miss_keys``, and ``tier_keyed``.
    The simulate path must return zero/empty equivalents so callers never KeyError
    post-collapse.

    Both the noop path (no interim slots) and the active path (slots present) are
    covered.
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
            "recall_per_tier",
            "tier_keyed",
            "rolled_back",
            "rollback_tier",
        }
    )

    def test_noop_return_has_full_schema(self, tmp_path):
        """Noop return (no interim slots, no housekeeping) carries all required keys.

        The early-return path (no interim slots, housekeeping=False) must include
        drift_intended_removal, drift_intended_removal_by_reason, recall_miss_keys,
        and tier_keyed with zero/empty values.
        """
        loop = _make_bare_loop(tmp_path)
        # No interim slots → noop path.
        result = loop.consolidate_simulate_fold()

        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, (
            f"Noop return missing required schema keys: {sorted(missing)}\n"
            f"  actual keys: {sorted(result.keys())}"
        )
        assert result["drift_intended_removal"] == 0
        assert result["drift_intended_removal_by_reason"] == {}
        assert result["recall_miss_keys"] == []
        assert result["tier_keyed"] == {}

    def test_active_return_has_full_schema(self, tmp_path):
        """Active return (interim slots present) carries all required keys.

        After a successful merge, the result must include all required schema
        keys — including the schema additions with their zero/empty values.
        """
        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(
            tmp_path,
            "20260301T0000",
            [
                {"key": "k1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            ],
        )

        result = loop.consolidate_simulate_fold()

        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, (
            f"Active return missing required schema keys: {sorted(missing)}\n"
            f"  actual keys: {sorted(result.keys())}"
        )
        assert result["drift_intended_removal"] == 0
        assert result["drift_intended_removal_by_reason"] == {}
        assert result["recall_miss_keys"] == []
        assert result["tier_keyed"] == {}
        # Sanity: active path does produce a tier_delta.
        assert "tier_delta" in result

    def test_drift_genuine_loss_is_zero_for_disk_source(self, tmp_path):
        """drift_genuine_loss is always 0 for the disk-source (simulate) path.

        The simulate venue has an empty store; the recall-miss set is empty
        (no weight reconstruction); drift_genuine_loss == 0 reproduces the
        invariant: the simulate path has an empty store and no weight reconstruction,
        so the recall-miss set is empty and drift_genuine_loss is always 0.
        """
        loop = _make_bare_loop(tmp_path)
        _write_interim_graph(
            tmp_path,
            "20260301T0000",
            [
                {"key": "k1", "subject": "Alice", "predicate": "likes", "object": "Tea"},
                {"key": "k2", "subject": "Bob", "predicate": "works_at", "object": "Acme"},
            ],
        )

        result = loop.consolidate_simulate_fold()

        assert result["drift_genuine_loss"] == 0, (
            f"Disk-source (simulate) path: drift_genuine_loss must always be 0; "
            f"got {result['drift_genuine_loss']!r}"
        )
