"""Tests for procedural facts in the B5 unified interim pipeline.

B5 routes procedural-typed facts through the same interim slot as episodic
facts, instead of training a separate per-cycle ``procedural`` MAIN adapter.

Key invariants verified here:
- A proc_graph merged into merger.graph reaches _tier_keyed["procedural"]
  (gap-regression: the proc_graph was previously extracted but never merged).
- Procedural interim keys are minted into the store only after successful
  training (deferred-write atomicity mirrors the episodic path).
- Simulate mode registers procedural interim keys immediately (mirrors episodic).
- B7 durability: a recall-failed NEW preference keeps its source session pending.
- The _run_indexed_key_procedural and _prepare_procedural_keys_for_tier
  per-cycle helper functions no longer exist (deleted in B5).
"""

from __future__ import annotations

import ast
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx

_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Structural guard: deleted per-cycle helpers must be absent
# ---------------------------------------------------------------------------


class TestDeletedHelpers:
    """The per-cycle procedural helpers deleted in B5 must not reappear."""

    def test_run_indexed_key_procedural_deleted(self):
        """_run_indexed_key_procedural must not exist in the AST."""
        src = (_PROJECT_ROOT / "paramem/training/consolidation.py").read_text()
        tree = ast.parse(src)
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "_run_indexed_key_procedural" not in func_names, (
            "_run_indexed_key_procedural was deleted in B5 and must not reappear"
        )

    def test_prepare_procedural_keys_for_tier_deleted(self):
        """_prepare_procedural_keys_for_tier must not exist in the AST."""
        src = (_PROJECT_ROOT / "paramem/training/consolidation.py").read_text()
        tree = ast.parse(src)
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "_prepare_procedural_keys_for_tier" not in func_names, (
            "_prepare_procedural_keys_for_tier was deleted in B5 and must not reappear"
        )


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_minimal_loop(tmp_path):
    """Return a ConsolidationLoop stub with enough state for B5 tests.

    Bypasses __init__ (object.__new__) to avoid model/GPU requirements.
    Mirrors the pattern used by TestAbortSkipsCommit._make_minimal_loop.
    """
    from peft import PeftModel

    from paramem.memory.store import MemoryStore
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    loop = object.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.model.__class__ = PeftModel
    loop.model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "procedural": MagicMock(),
        "in_training": MagicMock(),
    }
    loop.tokenizer = MagicMock()
    loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
    loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
    loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.procedural_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.wandb_config = None
    loop._thermal_policy = None
    loop.output_dir = tmp_path
    loop.store = MemoryStore(replay_enabled=True)
    loop.indexed_key_cache = {}
    loop.promoted_keys = set()
    loop.cycle_count = 0
    loop.episodic_simhash = {}
    loop.semantic_simhash = {}
    loop.procedural_simhash = {}
    loop._procedural_next_index = 0
    loop._procedural_tentative_next_index = 0
    loop.merger = MagicMock()
    loop.merger.graph.nodes = {}
    loop._bg_trainer = None
    loop.shutdown_requested = False
    loop._early_stop_callback = None
    loop.fingerprint_cache = None
    loop._keep_prior_slots = 2
    loop._debug_base = None
    loop.save_cycle_snapshots = False
    loop.snapshot_dir = None
    loop._indexed_next_index = 0
    loop._indexed_ep_interim = {}
    loop.episodic_replay_pool = []
    loop.curriculum_sampler = None
    return loop


def _common_patches(loop):
    """Return the context-manager list shared across cycle-level tests.

    Stubs out GPU-touching and PEFT-slot helpers that are not under test.
    """
    from paramem.training.consolidation import ConsolidationLoop

    return [
        patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
        patch(
            "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
            MagicMock,
        ),
        patch.object(
            ConsolidationLoop,
            "_resolve_target_slot",
            return_value="episodic_interim_t001",
        ),
        patch.object(ConsolidationLoop, "_refine_consolidation_graph", return_value=None),
        patch.object(ConsolidationLoop, "_enable_gradient_checkpointing", return_value=None),
        patch.object(ConsolidationLoop, "_disable_gradient_checkpointing", return_value=None),
        patch.object(ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)),
        patch("paramem.training.consolidation.switch_adapter"),
        patch(
            "paramem.training.consolidation.format_entry_training",
            return_value=[{"input_ids": [1], "labels": [1]}],
        ),
        patch(
            "paramem.memory.interim_adapter.create_interim_adapter",
            side_effect=lambda m, cfg, stamp: m,
        ),
    ]


# ---------------------------------------------------------------------------
# Test 1: proc_graph gap-regression (B5 core)
# ---------------------------------------------------------------------------


class TestProcGraphMergeGap:
    """Gap-regression: proc_graph merged into merger.graph reaches _tier_keyed["procedural"].

    Before B5 the proc_graph produced by run_procedural() was extracted but
    NEVER merged into merger.graph, so procedural-typed edges never reached
    _build_all_edge_entries_into's graph-walk.  This test proves the gap is
    closed: a keyless edge with relation_type="preference" in merger.graph
    ends up in _tier_keyed["procedural"] when _build_all_edge_entries_into
    is called with defer=True.
    """

    def test_proc_graph_edge_reaches_tier_keyed_procedural(self, tmp_path):
        """A procedural-typed keyless edge in merger.graph appears in _tier_keyed["procedural"]."""
        loop = _make_minimal_loop(tmp_path)

        # Simulate the result of merger.merge(proc_graph, ...):
        # one procedural-typed edge in merger.graph.
        g = nx.MultiDiGraph()
        g.add_node("alice", speaker_id="Speaker0", attributes={"name": "Alice"})
        g.add_node("tea", attributes={"name": "Tea"})
        g.add_edge("alice", "tea", predicate="prefers", relation_type="preference")
        loop.merger.graph = g

        tier_keyed: dict = {"episodic": [], "procedural": [], "semantic": []}
        _, deferred = loop._build_all_edge_entries_into(
            tier_keyed,
            default_speaker_id="Speaker0",
            defer=True,
            tag_new=True,
        )

        assert tier_keyed["procedural"], (
            "procedural-typed edge in merger.graph must appear in _tier_keyed['procedural']"
        )
        assert not tier_keyed["episodic"], "preference-typed edge must NOT land in episodic"
        # Exactly one deferred write for the procedural tier.
        proc_deferred = [r for r in deferred if r["tier"] == "procedural"]
        assert len(proc_deferred) == 1, (
            f"Expected 1 deferred procedural write; got {len(proc_deferred)}"
        )
        assert proc_deferred[0]["entry"]["key"].startswith("proc"), (
            f"Procedural key must start with 'proc'; got {proc_deferred[0]['entry']['key']}"
        )


# ---------------------------------------------------------------------------
# Test 2: procedural fact routed into interim slot, NOT procedural MAIN per cycle
# ---------------------------------------------------------------------------


class TestProceduralRoutedToInterim:
    """Per-cycle procedural training goes into the interim slot, not procedural MAIN.

    run_consolidation_cycle with a procedural-typed edge in merger.graph must
    call train_adapter on the interim adapter (episodic_interim_t001), never on
    the "procedural" MAIN adapter.
    """

    def test_procedural_edge_trains_on_interim_not_main(self, monkeypatch, tmp_path):
        """train_adapter is called with adapter_name=interim slot, not 'procedural'."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = _make_minimal_loop(tmp_path)

        g = nx.MultiDiGraph()
        g.add_node("bob", speaker_id="Speaker0", attributes={"name": "Bob"})
        g.add_node("jazz", attributes={"name": "Jazz"})
        g.add_edge("bob", "jazz", predicate="likes", relation_type="preference")
        loop.merger.graph = g

        train_calls: list[str] = []

        def _capture_train(**kwargs):
            train_calls.append(kwargs.get("adapter_name", "UNKNOWN"))
            return {"train_loss": 0.1, "aborted": False}

        patches = _common_patches(loop) + [
            patch("paramem.training.trainer.train_adapter", side_effect=_capture_train),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch.object(ConsolidationLoop, "_probe_passing_keys", return_value={"proc1"}),
        ]

        with _stack(patches):
            loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Bob",
                        "predicate": "likes",
                        "object": "Jazz",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test_interim_route",
                stamp="t001",
            )

        # Training must have fired exactly once, on the interim adapter.
        assert len(train_calls) == 1, f"Expected 1 train call; got {train_calls}"
        assert train_calls[0] == "episodic_interim_t001", (
            f"Procedural facts must train on interim slot; adapter was {train_calls[0]}"
        )


# ---------------------------------------------------------------------------
# Test 3: simulate mode registers procedural interim keys immediately
# ---------------------------------------------------------------------------


class TestSimulateModeRegistersProceduralKeys:
    """simulate mode applies deferred-flush for procedural keys immediately (no training).

    store.put must be called for the minted proc-key in simulate mode.
    The store tier for procedural keys in simulate mode is the interim slot
    (adapter_name), NOT "procedural" main — store tier must equal weight residence.
    """

    def test_simulate_mode_puts_proc_key_in_store(self, tmp_path):
        """simulate mode: minted proc-key appears in the interim store tier."""
        loop = _make_minimal_loop(tmp_path)
        loop.config.mode = "simulate"

        g = nx.MultiDiGraph()
        g.add_node("carol", speaker_id="Speaker0", attributes={"name": "Carol"})
        g.add_node("cycling", attributes={"name": "Cycling"})
        g.add_edge("carol", "cycling", predicate="enjoys", relation_type="preference")
        loop.merger.graph = g

        store_put_calls: list[tuple] = []
        original_put = loop.store.put

        def _spy_put(tier, key, entry, **kwargs):
            store_put_calls.append((tier, key))
            return original_put(tier, key, entry, **kwargs)

        from paramem.training.consolidation import ConsolidationLoop

        patches = [
            patch.object(
                ConsolidationLoop,
                "_resolve_target_slot",
                return_value="episodic_interim_t001",
            ),
            patch.object(ConsolidationLoop, "_refine_consolidation_graph", return_value=None),
            patch.object(ConsolidationLoop, "_enable_gradient_checkpointing", return_value=None),
            patch.object(ConsolidationLoop, "_disable_gradient_checkpointing", return_value=None),
            patch("paramem.training.consolidation.switch_adapter"),
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch.object(loop.store, "put", side_effect=_spy_put),
            patch("paramem.memory.persistence.commit_tier_slot"),
        ]

        with _stack(patches):
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Carol",
                        "predicate": "enjoys",
                        "object": "Cycling",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="simulate",
                run_label="test_simulate",
                stamp="t001",
            )

        # The proc-key must appear in the interim slot (adapter_name), not "procedural" main.
        # Store tier must equal weight residence: proc keys are trained into the interim
        # adapter, so they must be registered there.
        interim_slot = "episodic_interim_t001"
        proc_puts = [(t, k) for t, k in store_put_calls if k.startswith("proc")]
        assert proc_puts, "simulate mode must store.put procedural keys; no proc-prefix put found"
        for tier, key in proc_puts:
            assert tier == interim_slot, (
                f"Procedural key '{key}' must go into interim tier '{interim_slot}' in simulate; "
                f"got {tier!r} — store tier must equal weight residence"
            )

        assert result.get("mode") == "simulated", (
            f"Expected mode='simulated'; got {result.get('mode')}"
        )


# ---------------------------------------------------------------------------
# Test 4: B7 durability — recall-failed preference keeps session pending
# ---------------------------------------------------------------------------


class TestB7ProceduralSessionPending:
    """B7 durability: a recall-failed procedural key keeps its source session pending.

    session_ids carried on the edge (from merger.graph via proc_graph merge)
    must flow: proc_graph relation → edge["sessions"] → rec["session_ids"]
    → result["recall_failed_session_ids"] when training passes but recall check fails.

    The edge's ``sessions`` set is preserved through the materialize pass because
    the merger's ``_upsert_relation`` appends each Relation's ``session_ids`` onto
    the graph edge (merger.py:627-629).  The ``_SYNTHETIC_SESSION_IDS`` sentinel
    ``"__interim_pending_sessions__"`` is stripped by the deferred-write builder,
    leaving the real session ids.
    """

    def test_recall_failed_proc_key_session_kept_pending(self, tmp_path):
        """A proc-key whose training fails recall probe keeps its session_id pending."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = _make_minimal_loop(tmp_path)

        session_id = "session-proc-b7"
        g = nx.MultiDiGraph()
        g.add_node("dave", speaker_id="Speaker0", attributes={"name": "Dave"})
        g.add_node("hiking", attributes={"name": "Hiking"})
        g.add_edge(
            "dave",
            "hiking",
            predicate="loves",
            relation_type="preference",
            sessions={session_id},
        )
        loop.merger.graph = g

        # Train succeeds but probe admits NO keys (simulates recall failure).
        # Patching at class level so self._probe_passing_keys returns set() for
        # ALL keys, causing them to be excluded from store.put and their
        # session_ids to accumulate in _recall_failed_session_ids.
        patches = _common_patches(loop) + [
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.1, "aborted": False},
            ),
            patch("paramem.memory.persistence.commit_tier_slot"),
            # Probe passes nothing — all new keys "fail" recall.
            patch.object(ConsolidationLoop, "_probe_passing_keys", return_value=set()),
        ]

        with _stack(patches):
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Dave",
                        "predicate": "loves",
                        "object": "Hiking",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test_b7",
                stamp="t001",
            )

        # The session that produced the failing proc-key must be in keep-pending set.
        # recall_failed_session_ids is returned in the result dict (local variable,
        # not an instance attribute) — callers use result.get("recall_failed_session_ids", []).
        failed_ids = result.get("recall_failed_session_ids", [])
        assert session_id in failed_ids, (
            f"Session '{session_id}' must be in result['recall_failed_session_ids'] "
            f"after proc-key recall failure; got {failed_ids}"
        )


# ---------------------------------------------------------------------------
# Test 5: regression — proc keys land in INTERIM tier, not "procedural" main
# ---------------------------------------------------------------------------


class TestProceduralKeyRegisteredInInterimTier:
    """Regression: interim-cycle proc keys must be registered in the interim slot.

    Before the Finding-1 fix, deferred-flush wrote proc keys to the
    "procedural" MAIN store tier even though their weights were trained into
    the interim adapter.  The router pairs keys with the adapter named by
    their store tier, so the mismatch made those keys unrecallable for the
    entire interim window.

    This test FAILS on the old code (``_store_tier = "procedural"``).
    """

    def test_train_mode_proc_key_registered_in_interim_slot(self, tmp_path):
        """train mode: proc-key store tier == interim adapter name, not 'procedural'."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = _make_minimal_loop(tmp_path)

        g = nx.MultiDiGraph()
        g.add_node("eve", speaker_id="Speaker0", attributes={"name": "Eve"})
        g.add_node("running", attributes={"name": "Running"})
        g.add_edge("eve", "running", predicate="enjoys", relation_type="preference")
        loop.merger.graph = g

        store_put_calls: list[tuple] = []
        original_put = loop.store.put

        def _spy_put(tier, key, entry, **kwargs):
            store_put_calls.append((tier, key))
            return original_put(tier, key, entry, **kwargs)

        interim_slot = "episodic_interim_t001"
        patches = _common_patches(loop) + [
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.05, "aborted": False},
            ),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch.object(ConsolidationLoop, "_probe_passing_keys", return_value={"proc0"}),
            patch.object(loop.store, "put", side_effect=_spy_put),
        ]

        with _stack(patches):
            loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Eve",
                        "predicate": "enjoys",
                        "object": "Running",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test_tier_regression",
                stamp="t001",
            )

        proc_puts = [(t, k) for t, k in store_put_calls if k.startswith("proc")]
        assert proc_puts, "train mode must store.put the proc key after successful training"
        for tier, key in proc_puts:
            assert tier == interim_slot, (
                f"Proc key '{key}' registered in tier '{tier}'; expected '{interim_slot}'. "
                "Store tier must equal weight residence (the interim adapter)."
            )

        # Also verify that the key is NOT registered in the "procedural" main tier.
        proc_main_puts = [(t, k) for t, k in store_put_calls if t == "procedural"]
        assert not proc_main_puts, (
            f"No key must be stored in 'procedural' main during an interim cycle; "
            f"got: {proc_main_puts}"
        )

    def test_train_mode_proc_key_has_preference_bookkeeping(self, tmp_path):
        """bookkeeping relation_type is 'preference' so COMMAND filter classifies it correctly."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = _make_minimal_loop(tmp_path)

        g = nx.MultiDiGraph()
        g.add_node("frank", speaker_id="Speaker0", attributes={"name": "Frank"})
        g.add_node("chess", attributes={"name": "Chess"})
        g.add_edge("frank", "chess", predicate="plays", relation_type="preference")
        loop.merger.graph = g

        bk_calls: list[dict] = []
        original_bk = loop.store.set_bookkeeping

        def _spy_bk(key, **kwargs):
            bk_calls.append({"key": key, **kwargs})
            return original_bk(key, **kwargs)

        patches = _common_patches(loop) + [
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.05, "aborted": False},
            ),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch.object(ConsolidationLoop, "_probe_passing_keys", return_value={"proc0"}),
            patch.object(loop.store, "set_bookkeeping", side_effect=_spy_bk),
        ]

        with _stack(patches):
            loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Frank",
                        "predicate": "plays",
                        "object": "Chess",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test_bk_preference",
                stamp="t001",
            )

        proc_bk = [c for c in bk_calls if c["key"].startswith("proc")]
        assert proc_bk, "set_bookkeeping must be called for the proc key"
        for call in proc_bk:
            assert call.get("relation_type") == "preference", (
                f"Proc key bookkeeping must carry relation_type='preference'; "
                f"got {call.get('relation_type')!r}"
            )

    def test_interim_active_keys_includes_proc_key(self, tmp_path):
        """After an interim cycle, proc key is active in the interim tier (not procedural main).

        This is the router-level regression: active_keys_in_tier(interim_slot) returns
        the proc key; active_keys_in_tier("procedural") does NOT.
        """
        from paramem.training.consolidation import ConsolidationLoop

        loop = _make_minimal_loop(tmp_path)

        g = nx.MultiDiGraph()
        g.add_node("gwen", speaker_id="Speaker0", attributes={"name": "Gwen"})
        g.add_node("yoga", attributes={"name": "Yoga"})
        g.add_edge("gwen", "yoga", predicate="practices", relation_type="preference")
        loop.merger.graph = g

        interim_slot = "episodic_interim_t001"
        patches = _common_patches(loop) + [
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.05, "aborted": False},
            ),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch.object(ConsolidationLoop, "_probe_passing_keys", return_value={"proc0"}),
        ]

        with _stack(patches):
            loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Gwen",
                        "predicate": "practices",
                        "object": "Yoga",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test_active_keys",
                stamp="t001",
            )

        interim_active = list(loop.store.active_keys_in_tier(interim_slot))
        proc_main_active = list(loop.store.active_keys_in_tier("procedural"))

        assert any(k.startswith("proc") for k in interim_active), (
            f"Proc key must be active in interim tier '{interim_slot}'; "
            f"active keys there: {interim_active}"
        )
        assert not any(k.startswith("proc") for k in proc_main_active), (
            f"Proc key must NOT appear in 'procedural' main during an interim cycle; "
            f"active keys there: {proc_main_active}"
        )


# ---------------------------------------------------------------------------
# Helper: stack multiple context managers from a list
# ---------------------------------------------------------------------------


def _stack(patches):
    """Enter all patches in the list; return a single combined context manager."""

    class _MultiCtx:
        def __enter__(self):
            self._stack = ExitStack()
            for p in patches:
                self._stack.enter_context(p)
            return self

        def __exit__(self, *exc):
            return self._stack.__exit__(*exc)

    return _MultiCtx()
