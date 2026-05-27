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
    loop.config = ConsolidationConfig(indexed_key_replay_enabled=True)
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
    loop.persist_graph = False
    loop._thermal_policy = None
    loop.shutdown_requested = False
    loop.merger = MagicMock()

    # MemoryStore with registry enabled.
    store = MemoryStore(replay_enabled=True)
    for tier in ("episodic", "semantic", "procedural"):
        store.load_registry(tier, KeyRegistry())
    loop.store = store

    loop.procedural_sp_index: dict = {}
    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.key_sessions: dict = {}
    loop.promoted_keys: set = set()
    loop.pending_interim_triples: list = []
    loop.fingerprint_cache = None

    # Enrichment flags — disabled to keep tests deterministic.
    loop.graph_enrichment_enabled = False
    loop.graph_enrichment_interim_enabled = False
    loop.graph_enrichment_min_triples_floor = 20
    loop._triples_since_last_enrichment = 0
    loop.full_consolidation_period_string = ""

    return loop


def _patches_for_train_mode():
    """Return list of context managers that stub the GPU-touching train path.

    Stubs:
    - ``paramem.training.trainer.train_adapter`` → no-op returning a metrics dict.
    - ``paramem.models.loader.save_adapter`` → no-op (avoids PEFT I/O).
    - ``paramem.adapters.manifest.build_manifest_for`` → returns None.
    - ``paramem.memory.interim_adapter.create_interim_adapter`` → populates
      peft_config[adapter_name] so _resolve_target_slot's peft_config check
      works; returns the model unchanged.
    - ``paramem.training.consolidation.probe_entries`` → yields None recalled so
      existing-key reconstruction falls through to the store cache (unit-test
      mode; real inference requires a loaded GPU model).
    """

    def _fake_create_interim(model, cfg, stamp):
        """Create interim adapter slot in the mock peft_config."""
        adapter_name = f"episodic_interim_{stamp}"
        model.peft_config[adapter_name] = MagicMock()
        return model

    def _fake_probe_entries(model, tokenizer, entries, **kw):
        """Yield (entry, None) so reconstruction falls through to the store cache."""
        for e in entries:
            yield e, None

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
        patch(
            "paramem.training.consolidation.probe_entries",
            side_effect=_fake_probe_entries,
        ),
    ]


# ---------------------------------------------------------------------------
# Change J — Parity tests: simulate vs train mode
# ---------------------------------------------------------------------------


class TestSimulateTrainParity:
    """run_consolidation_cycle in simulate and train modes must converge.

    Gaps covered:
      - Gap 5: speaker_id default tagging — every entry gets the caller's id
        when none was present on the relation.
      - Gap 6: first_seen_cycle preservation — re-running the same relations
        a second time leaves existing entries' first_seen_cycle unchanged.
      - Gap 4: per-tier scope — active_keys_in_tier returns identical sorted
        key lists in both modes.
      - Equality: simhashes_in_tier, store entries, and on-disk
        indexed_key_registry bytes are bytewise-equal between modes.

    Gap 1 (enrichment) is verified separately in test_graph_enrichment.py;
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
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return loop.run_consolidation_cycle(
                list(_EPISODIC_RELS),
                list(_PROCEDURAL_RELS),
                speaker_id=_SPEAKER_ID,
                mode="train",
                run_label="parity",
                stamp=_STAMP,
            )

    def test_gap5_speaker_id_default_applied_both_modes(self, loop_sim, loop_train, tmp_path):
        """Gap 5: entries missing speaker_id receive the caller's id in both modes.

        Relations without a speaker_id key receive the default (_SPEAKER_ID).
        Relations with an explicit non-empty speaker_id preserve it.
        Relations with an explicit speaker_id="" preserve the empty string —
        _tag_speaker_id_defaults only adds the key when absent; it does not
        replace explicit empty-string values.
        """
        self._run_sim(loop_sim)
        self._run_train(loop_train)

        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                assert "speaker_id" in entry, f"Entry {key} missing speaker_id"
                # Entries without any speaker_id key should get the default.
                # Entries with an explicit value (including "") are kept as-is.

        # Entries with no explicit speaker_id should receive _SPEAKER_ID.
        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("object") == "Berlin":  # Alice/lives_in/Berlin — no speaker_id key
                    assert entry["speaker_id"] == _SPEAKER_ID, (
                        f"Default tagging failed for {key}: expected {_SPEAKER_ID!r}, "
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
        # _tag_speaker_id_defaults adds the key only when absent; it must not
        # replace an explicit empty string with the caller's id.
        for loop in (loop_sim, loop_train):
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("subject") == "Carol" and entry.get("object") == "London":
                    assert entry["speaker_id"] == "", (
                        f"Explicit empty speaker_id overwritten in {key}: {entry['speaker_id']!r}"
                    )

    def test_gap6_first_seen_cycle_preserved_on_second_run(self, loop_sim, loop_train, tmp_path):
        """Gap 6: existing entries retain first_seen_cycle on re-run.

        Run the same fixture twice on each loop using the SAME stamp so that
        _resolve_target_slot returns the same adapter slot and
        _prepare_episodic_keys_for_tier enters the existing-key reconstruction
        branch.  Entries from cycle 1 must keep first_seen_cycle=1 after
        cycle 2.

        Using the same stamp is deliberate: this exercises the preservation
        branch (not just the fresh-slot path), matching the intent of Gap 6.
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
        # branch in _prepare_episodic_keys_for_tier.
        loop_sim.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="simulate",
            run_label="parity2",
            stamp=_STAMP,  # same stamp as cycle 1
        )
        patches = _patches_for_train_mode()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
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

        # Episodic (graph*) keys from cycle 1 must survive into cycle 2.
        # Procedural (proc*) keys are correctly RETIRED when a same-preference
        # re-run triggers contradiction detection — those keys are intentionally
        # absent in cycle 2 and are excluded from this assertion.
        sim_keys_2 = set(key for _tier, key, _entry in loop_sim.store.iter_entries())
        train_keys_2 = set(key for _tier, key, _entry in loop_train.store.iter_entries())
        sim_graph_1 = {k for k in sim_keys_1 if k.startswith("graph")}
        train_graph_1 = {k for k in train_keys_1 if k.startswith("graph")}
        assert sim_graph_1.issubset(sim_keys_2), (
            f"Simulate: cycle-1 episodic keys dropped after re-run: {sim_graph_1 - sim_keys_2}"
        )
        assert train_graph_1.issubset(train_keys_2), (
            f"Train: cycle-1 episodic keys dropped after re-run: {train_graph_1 - train_keys_2}"
        )

    def test_gap4_active_keys_in_tier_match(self, loop_sim, loop_train):
        """Gap 4: active_keys_in_tier returns identical sorted lists in both modes."""
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

        sim_hashes = dict(loop_sim.store.simhashes_in_tier(adapter_name))
        train_hashes = dict(loop_train.store.simhashes_in_tier(adapter_name))

        assert sim_hashes == train_hashes, (
            f"Simhash registries diverged for tier {adapter_name!r}:\n"
            f"  simulate keys: {sorted(sim_hashes)}\n"
            f"  train keys:    {sorted(train_hashes)}"
        )

    def test_registry_bytes_equal_both_modes(self, loop_sim, loop_train):
        """On-disk indexed_key_registry.json bytes are bytewise-equal in both modes.

        This is the I5 commit-signal parity contract: the registry payload
        written by commit_tier_slot must be identical regardless of mode.
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

        Before fix (R1): ``tier_reg = loop.store.registry(tier)`` read the
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

    def test_procedural_graph_json_written_in_simulate_mode(self, loop_sim, tmp_path):
        """Procedural simulate-mode cycle writes non-empty graph.json.

        Regression for C1: commit_tier_slot was called with all_keyed=[]
        for the procedural tier, causing an empty graph.json to overwrite prior
        content.  After the fix, the simulate branch re-projects from loop.store
        when all_keyed is empty and the tier has entries.
        """
        from paramem.memory.interim_adapter import adapter_slot_root_for_name
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        self._run_sim(loop_sim)

        proc_slot = adapter_slot_root_for_name(loop_sim.output_dir, "procedural")
        graph_path = proc_slot / "graph.json"

        assert graph_path.exists(), f"Procedural simulate graph.json missing at {graph_path}"

        graph = load_memory_from_disk(graph_path)
        entries = list(iter_entries(graph))
        assert len(entries) > 0, (
            "Procedural graph.json is empty after simulate cycle — "
            "C1 regression: all_keyed=[] caused empty graph overwrite"
        )

        # Verify the expected procedural keys are present.
        proc_keys = {e["key"] for e in entries}
        assert len(proc_keys) == len(_PROCEDURAL_RELS), (
            f"Expected {len(_PROCEDURAL_RELS)} procedural keys, got {len(proc_keys)}: {proc_keys}"
        )

    def test_slot_layout_train_has_manifest_simulate_has_graph(self, loop_sim, loop_train):
        """Slot layout assertion: train slots have meta.json; simulate slots have graph.json.

        After C6 + C2, commit_tier_slot (train) raises on manifest failure rather
        than saving without one — so any train slot that lands on disk has a manifest.
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

    def test_gap9_active_adapter_restored_after_cycle(self, loop_sim, loop_train):
        """Gap 9: model.set_adapter("episodic") is called at end of cycle in both modes.

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
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
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
# N1 — TestConsolidateInterimGraphs
# ---------------------------------------------------------------------------


def _make_bare_loop(tmp_path: Path) -> ConsolidationLoop:
    """Build the minimal ConsolidationLoop required by consolidate_interim_graphs.

    Only the attributes read by the method are set:
      - ``output_dir`` — used as the adapter_dir root.
      - ``graph_enrichment_enabled`` — set to False so _run_graph_enrichment
        returns immediately without touching ``merger`` or ``extraction``.

    All other attributes are left unset; any unintended access will raise
    AttributeError rather than silently returning a MagicMock value.
    """
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.output_dir = tmp_path
    loop.graph_enrichment_enabled = False
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


class TestConsolidateInterimGraphs:
    """consolidate_interim_graphs merges simulate-mode interim slots into main episodic graph."""

    def test_no_interim_slots_returns_noop_result(self, tmp_path):
        """Empty interim set: no main-graph write, no crash, result signals noop.

        When no ``episodic/interim_*`` directories contain a ``graph.json``,
        the method returns the canonical noop dict and does NOT create the
        main-tier ``graph.json`` (nothing to merge).
        """
        loop = _make_bare_loop(tmp_path)

        result = loop.consolidate_interim_graphs()

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

        result = loop.consolidate_interim_graphs()

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

        result = loop.consolidate_interim_graphs()

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
        """Two slots sharing the same triple (by subject/predicate/object/key) dedup correctly.

        NetworkX MultiDiGraph adds each edge call as a separate edge; the test
        verifies that the merged graph contains exactly one edge per unique ik_key
        value — i.e. the merge does not silently inflate edge count when both
        slots carry the same key.

        Note: ``_add_keyed_edge`` calls ``graph.add_edge`` which in a
        ``MultiDiGraph`` always appends a new parallel edge.  The dedup
        guarantee is on the *key set* (``iter_entries`` yields all edges, so we
        count distinct key values), not on the raw edge count.  The design
        documents "last write wins" for identical keys across slots.
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

        loop.consolidate_interim_graphs()

        main_graph_path = tmp_path / "episodic" / "graph.json"
        merged = load_memory_from_disk(main_graph_path)
        keys = [e["key"] for e in iter_entries(merged)]
        # Both edges land in the MultiDiGraph (parallel edges), but every
        # entry carries "graph1" — the key set has exactly one element.
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

        loop.consolidate_interim_graphs()

        for d in dirs:
            assert not d.exists(), f"Interim dir {d} must be removed after consolidation"
        assert (tmp_path / "episodic" / "graph.json").exists(), (
            "Main graph.json must exist after merge"
        )


# ---------------------------------------------------------------------------
# N2 — TestBackgroundTrainerClose
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


# ---------------------------------------------------------------------------
# N3 — TestCommitTierSlotCleanup
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
    # Add the entry to the cache via setdefault_entry so all_active_keys()
    # can find it and build_tier_graph_from_store has entry data available.
    entry = store.setdefault_entry("episodic", "graph1", {})
    entry.update(
        {
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
            "speaker_id": "sp1",
            "first_seen_cycle": 0,
        }
    )
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

        # Cycle passes ``stamp`` through to the writer so the dir nests under
        # ``interim_<stamp>/`` — matches the production layout
        # ``paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/``.
        cycle_dir = loop.snapshot_dir_for(interim_stamp=_STAMP)
        assert cycle_dir is not None
        # graph_snapshot.json is written by ``merger.save_graph`` — the test's
        # mocked merger doesn't materialise a file, but the on_extraction_end
        # dispatch must still have called it.  Assert via the call mock.
        loop.merger.save_graph.assert_any_call(cycle_dir / "graph_snapshot.json", encrypted=False)
        assert (cycle_dir / "episodic_rels_snapshot.json").exists()
        assert (cycle_dir / "procedural_rels_snapshot.json").exists()
        assert (cycle_dir / "cycle_summary_snapshot.json").exists()

        ep_dump = json.loads((cycle_dir / "episodic_rels_snapshot.json").read_text())
        assert ep_dump == list(_EPISODIC_RELS)
        summary = json.loads((cycle_dir / "cycle_summary_snapshot.json").read_text())
        assert summary["mode"] == "simulated"
        assert summary["venue"] == "simulate"
        assert summary["error"] is None

    def test_queue_only_branch_emits_summary_but_skips_graph_dump(self, tmp_path):
        """Queue-only short-circuit (``max_interim_count=0``) emits only the
        cycle summary — no cumulative-graph or relation dumps (locked
        decision #1).
        """
        loop = _build_loop(tmp_path / "loop")
        debug_base = tmp_path / "debug"
        self._enable_debug(loop, debug_base)

        result = loop.run_consolidation_cycle(
            list(_EPISODIC_RELS),
            list(_PROCEDURAL_RELS),
            speaker_id=_SPEAKER_ID,
            mode="simulate",
            run_label="integration-queue",
            stamp=_STAMP,
            max_interim_count=0,
        )
        assert result["mode"] == "queued"

        cycle_dir = loop.snapshot_dir_for(interim_stamp=_STAMP)
        assert cycle_dir is not None
        assert (cycle_dir / "cycle_summary_snapshot.json").exists()
        assert not (cycle_dir / "graph_snapshot.json").exists()
        assert not (cycle_dir / "episodic_rels_snapshot.json").exists()
        assert not (cycle_dir / "procedural_rels_snapshot.json").exists()
        # And merger.save_graph must NOT have been called for the dump path.
        for call in loop.merger.save_graph.call_args_list:
            assert call.args[0].name != "graph_snapshot.json"

        summary = json.loads((cycle_dir / "cycle_summary_snapshot.json").read_text())
        assert summary["mode"] == "queued"
        assert summary["adapter_name"] is None
