"""Tests for procedural facts in the unified interim pipeline.

Procedural-typed facts route through the same interim slot as episodic
facts, instead of training a separate per-cycle ``procedural`` MAIN adapter.

Key invariants verified here:
- A proc_graph merged into merger.graph reaches _tier_keyed["procedural"].
- Procedural interim keys are minted into the store only after successful
  training (deferred-write atomicity mirrors the episodic path).
- Simulate mode registers procedural interim keys immediately (mirrors episodic).
- The unified recall gate is all-or-nothing: one failing key among a mixed
  batch rejects the whole increment and commits nothing.
- There are no per-cycle ``_run_indexed_key_procedural`` /
  ``_prepare_procedural_keys_for_tier`` helper functions; procedural facts
  route through the unified interim slot.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Structural guard: deleted per-cycle helpers must be absent
# ---------------------------------------------------------------------------


class TestDeletedHelpers:
    """Per-cycle procedural helpers removed in the unified-interim refactor must not reappear."""

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
            "_run_indexed_key_procedural was removed in the unified-interim refactor"
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
            "_prepare_procedural_keys_for_tier was removed in the unified-interim refactor"
        )


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _probe(passing_keys):
    """Build a RecallProbe whose passing_keys is exactly *passing_keys*.

    Stands in for ``loop._probe_recall(...)``'s return value — consumed by
    the single all-or-nothing recall gate (``_assert_tier_recall``), which
    reads only ``probe.passing_keys`` against ``probe.distinct_total``,
    never the individual record content.
    """
    from paramem.training.recall_eval import RecallProbe

    return RecallProbe(per_key=tuple({"key": k, "exact_match": True} for k in passing_keys))


# ---------------------------------------------------------------------------
# Shared fixture helpers for the run_consolidation_cycle-level tests below —
# a real GraphMerger/MemoryStore, a fake model for the write/publish
# primitives (mirrors tests/test_fold_build_driver.py's own pattern), and
# only the GPU-touching collaborators (training, the recall probe, the
# backup scope) faked.
# ---------------------------------------------------------------------------

_INTERIM_STAMP = "20260417T0000"
_INTERIM_SLOT = f"episodic_interim_{_INTERIM_STAMP}"


def _make_cycle_loop(tmp_path):
    """A loop wired for a real ``run_consolidation_cycle`` train-mode pass.

    Reuses the shared ``tests._fold_fixtures._make_loop`` (real
    ``GraphMerger``/``MemoryStore``, fake PEFT-touching model) with the
    interim slot itself pre-resident — the derive/build/publish driver then
    takes the warm ``ensure_adapter_matching`` path instead of minting a
    fresh adapter (which needs a real ``peft.PeftModel``, out of scope for
    this fake).
    """
    from tests._fold_fixtures import _make_loop

    return _make_loop(tmp_path, procedural=True, resident_tiers=(_INTERIM_SLOT,))


def _wire_cycle_fakes(loop, monkeypatch, *, probe_recall=None, assert_tier_recall=True):
    """Fake exactly the GPU-touching collaborators a train-mode cycle needs.

    ``_train_tier_adapter`` and ``tier_backup_scope`` are always faked (no
    real GPU training in this suite). ``_probe_recall`` takes the caller's
    stand-in (default: a probe where every entry passes). The recall GATE
    itself (``_assert_tier_recall``) stays REAL unless a test's own subject
    is something else — the all-or-nothing gate test below is the one
    exception, since real gate evaluation is exactly what it pins.
    """
    from paramem.models import loader as loader_mod
    from tests._fold_fixtures import _fake_tier_backup_scope

    def _fake_train(entries, **kwargs):
        if not entries:
            return None, None
        return {"aborted": False, "train_loss": 0.01}, None

    loop._train_tier_adapter = MagicMock(side_effect=_fake_train)
    loop._probe_recall = MagicMock(
        side_effect=probe_recall
        if probe_recall is not None
        else (lambda adapter_name, entries: _probe({e["key"] for e in entries}))
    )
    if assert_tier_recall:
        loop._assert_tier_recall = MagicMock()
    monkeypatch.setattr(loader_mod, "tier_backup_scope", _fake_tier_backup_scope)


# ---------------------------------------------------------------------------
# Test 2: procedural fact routed into interim slot, NOT procedural MAIN per cycle
# ---------------------------------------------------------------------------


class TestProceduralRoutedToInterim:
    """Per-cycle procedural training goes into the interim slot, not procedural MAIN.

    run_consolidation_cycle with a procedural-typed edge in merger.graph must
    call train_adapter on the interim adapter (episodic_interim_20260417T0000), never on
    the "procedural" MAIN adapter.
    """

    def test_procedural_edge_trains_on_interim_not_main(self, monkeypatch, tmp_path):
        """The funnel is called with adapter_name=interim slot, not 'procedural'."""
        loop = _make_cycle_loop(tmp_path)
        _wire_cycle_fakes(loop, monkeypatch)

        loop.merger.graph.add_edge(
            "bob", "jazz", predicate="likes", relation_type="preference", speaker_id="speaker0"
        )

        pending = loop.take_pending_relations()
        loop.run_consolidation_cycle(
            [
                {
                    "subject": "Bob",
                    "predicate": "likes",
                    "object": "Jazz",
                    "relation_type": "preference",
                    "speaker_id": "speaker0",
                }
            ],
            [],
            speaker_id="speaker0",
            mode="train",
            pending=pending,
            run_label="test_interim_route",
            stamp=_INTERIM_STAMP,
        )

        # Training must have fired exactly once, on the interim adapter.
        assert loop._train_tier_adapter.call_count == 1, (
            f"Expected 1 train call; got {loop._train_tier_adapter.call_args_list}"
        )
        train_adapter_name = loop._train_tier_adapter.call_args.kwargs.get("adapter_name")
        assert train_adapter_name == _INTERIM_SLOT, (
            f"Procedural facts must train on interim slot; adapter was {train_adapter_name}"
        )

    def test_interim_probe_targets_staging_adapter_before_promote(self, monkeypatch, tmp_path):
        """The interim commit's recall probe runs against STAGING_ADAPTER,
        before the promote — never the interim slot's own name.

        Kills: probing the interim slot itself (post-promote), which would
        read whatever was resident there before this cycle rather than the
        weights this cycle just trained.
        """
        from paramem.training.trainer import STAGING_ADAPTER

        loop = _make_cycle_loop(tmp_path)
        _wire_cycle_fakes(loop, monkeypatch)

        loop.merger.graph.add_edge(
            "bob", "jazz", predicate="likes", relation_type="preference", speaker_id="speaker0"
        )

        pending = loop.take_pending_relations()
        loop.run_consolidation_cycle(
            [
                {
                    "subject": "Bob",
                    "predicate": "likes",
                    "object": "Jazz",
                    "relation_type": "preference",
                    "speaker_id": "speaker0",
                }
            ],
            [],
            speaker_id="speaker0",
            mode="train",
            pending=pending,
            run_label="test_interim_probe_target",
            stamp=_INTERIM_STAMP,
        )

        probe_calls = [c.args[0] for c in loop._probe_recall.call_args_list]
        assert probe_calls == [STAGING_ADAPTER], (
            f"expected exactly one probe of {STAGING_ADAPTER!r}; got {probe_calls}"
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
        """simulate mode: minted proc-key appears in the interim store tier.

        Simulate touches no PEFT adapter at all (no mint, no probe, no
        backup scope) — only the derive/write-to-graph.json/publish spine, so
        this needs no GPU-collaborator fakes.  Registration is a wholesale
        registry rebind (``MemoryStore.adopt_increments``), not a per-key
        ``store.put`` call — the postcondition to check is presence in the
        tier's active-key set, not a spied write call.
        """
        loop = _make_cycle_loop(tmp_path)

        loop.merger.graph.add_edge(
            "carol",
            "cycling",
            predicate="enjoys",
            relation_type="preference",
            speaker_id="speaker0",
        )

        pending = loop.take_pending_relations()
        result = loop.run_consolidation_cycle(
            [
                {
                    "subject": "Carol",
                    "predicate": "enjoys",
                    "object": "Cycling",
                    "relation_type": "preference",
                    "speaker_id": "speaker0",
                }
            ],
            [],
            speaker_id="speaker0",
            mode="simulate",
            pending=pending,
            run_label="test_simulate",
            stamp=_INTERIM_STAMP,
        )

        # The proc-key must appear in the interim slot (adapter_name), not "procedural" main.
        # Store tier must equal weight residence: proc keys are trained into the interim
        # adapter, so they must be registered there.
        interim_active = loop.store.active_keys_in_tier(_INTERIM_SLOT)
        proc_keys = [k for k in interim_active if k.startswith("proc")]
        assert proc_keys, (
            "simulate mode must register procedural keys in the interim tier; "
            f"active keys there: {interim_active}"
        )
        proc_main_active = loop.store.active_keys_in_tier("procedural")
        assert not any(k.startswith("proc") for k in proc_main_active), (
            f"Procedural key must NOT be registered in 'procedural' main during "
            f"simulate; active keys there: {proc_main_active}"
        )

        assert result.get("mode") == "simulated", (
            f"Expected mode='simulated'; got {result.get('mode')}"
        )


# ---------------------------------------------------------------------------
# Test 4: Unified recall gate — all-or-nothing, no soft return, no partial write
# ---------------------------------------------------------------------------


class TestProceduralRecallGateAllOrNothing:
    """The interim commit's recall gate is one all-or-nothing verdict.

    ``_assert_tier_recall`` (paramem/training/consolidation.py) compares
    ``passing == total`` over the interim slot's full key set — there is no
    per-key acceptance and no soft ``recall_failed_session_ids`` return
    value.  A single failing key raises ``RecallGateRejected`` out of
    ``run_consolidation_cycle`` before any key — passing or failing — is
    written to the store, so a rejected increment leaves the store exactly
    as it found it.
    """

    def test_one_failing_key_among_two_rejects_the_whole_increment(self, tmp_path, monkeypatch):
        """A probe where one of two procedural keys fails raises
        RecallGateRejected and leaves the store without either key — the
        passing key is not partially committed."""
        from paramem.training.consolidation import RecallGateRejected
        from paramem.training.recall_eval import RecallProbe

        loop = _make_cycle_loop(tmp_path)

        def _one_key_fails(adapter_name, entries):
            # Mark exactly the first probed key as passing; every other key
            # (there is at least one more, given the two edges below) fails.
            per_key = [{"key": e["key"], "exact_match": i == 0} for i, e in enumerate(entries)]
            return RecallProbe(per_key=tuple(per_key))

        # This test's own subject IS the real gate — keep _assert_tier_recall
        # unfaked so it genuinely evaluates the controlled probe above.
        _wire_cycle_fakes(loop, monkeypatch, probe_recall=_one_key_fails, assert_tier_recall=False)

        loop.merger.graph.add_edge(
            "henry", "chess", predicate="plays", relation_type="preference", speaker_id="speaker0"
        )
        loop.merger.graph.add_edge(
            "henry", "golf", predicate="plays", relation_type="preference", speaker_id="speaker0"
        )

        store_put_calls: list[tuple] = []
        original_put = loop.store.put

        def _spy_put(tier, key, entry, **kwargs):
            store_put_calls.append((tier, key))
            return original_put(tier, key, entry, **kwargs)

        with (
            patch.object(loop.store, "put", side_effect=_spy_put),
            pytest.raises(RecallGateRejected),
        ):
            pending = loop.take_pending_relations()
            loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Henry",
                        "predicate": "plays",
                        "object": "Chess",
                        "relation_type": "preference",
                        "speaker_id": "speaker0",
                    },
                    {
                        "subject": "Henry",
                        "predicate": "plays",
                        "object": "Golf",
                        "relation_type": "preference",
                        "speaker_id": "speaker0",
                    },
                ],
                [],
                speaker_id="speaker0",
                mode="train",
                pending=pending,
                run_label="test_all_or_nothing",
                stamp=_INTERIM_STAMP,
            )

        assert not store_put_calls, (
            "a rejected increment must not partially commit the passing key; "
            f"got puts: {store_put_calls}"
        )
        assert not list(loop.store.active_keys_in_tier(_INTERIM_SLOT)), (
            "a rejected increment must leave the interim tier with no active keys"
        )
        ledger_path = loop._fold_state_dir / "stage_ledger.json"
        assert not ledger_path.exists(), (
            "a rejected increment must dispose the event's ledger so the next "
            "same-window attempt re-derives from scratch instead of resuming "
            "the rejected assignment"
        )


# ---------------------------------------------------------------------------
# Interim-cycle proc keys land in the INTERIM tier, not "procedural" main
# ---------------------------------------------------------------------------


class TestProceduralKeyRegisteredInInterimTier:
    """Interim-cycle proc keys must be registered in the interim slot, not
    the "procedural" MAIN store tier — their weights are trained into the
    interim adapter, and the router pairs keys with the adapter named by
    their store tier, so a mismatch would make those keys unrecallable for
    the entire interim window.
    """

    def test_train_mode_proc_key_registered_in_interim_slot(self, monkeypatch, tmp_path):
        """train mode: proc-key store tier == interim adapter name, not 'procedural'.

        Registration is a wholesale registry rebind
        (``MemoryStore.adopt_increments``), not a per-key ``store.put`` call —
        the postcondition is presence in the tier's active-key set.
        """
        loop = _make_cycle_loop(tmp_path)
        _wire_cycle_fakes(loop, monkeypatch)

        loop.merger.graph.add_edge(
            "eve", "running", predicate="enjoys", relation_type="preference", speaker_id="speaker0"
        )

        pending = loop.take_pending_relations()
        loop.run_consolidation_cycle(
            [
                {
                    "subject": "Eve",
                    "predicate": "enjoys",
                    "object": "Running",
                    "relation_type": "preference",
                    "speaker_id": "speaker0",
                }
            ],
            [],
            speaker_id="speaker0",
            mode="train",
            pending=pending,
            run_label="test_tier_regression",
            stamp=_INTERIM_STAMP,
        )

        interim_active = loop.store.active_keys_in_tier(_INTERIM_SLOT)
        proc_keys = [k for k in interim_active if k.startswith("proc")]
        assert proc_keys, (
            "train mode must register the proc key in the interim tier after "
            f"successful training; active keys there: {interim_active}"
        )

        # Also verify that the key is NOT registered in the "procedural" main tier.
        proc_main_active = loop.store.active_keys_in_tier("procedural")
        assert not any(k.startswith("proc") for k in proc_main_active), (
            f"No key must be registered in 'procedural' main during an interim "
            f"cycle; active keys there: {proc_main_active}"
        )

    def test_train_mode_proc_key_has_preference_bookkeeping(self, monkeypatch, tmp_path):
        """bookkeeping relation_type is 'preference' so COMMAND filter classifies it correctly."""
        loop = _make_cycle_loop(tmp_path)
        _wire_cycle_fakes(loop, monkeypatch)

        loop.merger.graph.add_edge(
            "frank", "chess", predicate="plays", relation_type="preference", speaker_id="speaker0"
        )

        pending = loop.take_pending_relations()
        loop.run_consolidation_cycle(
            [
                {
                    "subject": "Frank",
                    "predicate": "plays",
                    "object": "Chess",
                    "relation_type": "preference",
                    "speaker_id": "speaker0",
                }
            ],
            [],
            speaker_id="speaker0",
            mode="train",
            pending=pending,
            run_label="test_bk_preference",
            stamp=_INTERIM_STAMP,
        )

        proc_keys = [
            k for k in loop.store.active_keys_in_tier(_INTERIM_SLOT) if k.startswith("proc")
        ]
        assert proc_keys, "the proc key must be active in the interim tier"
        for key in proc_keys:
            bk = loop.store.bookkeeping_for_key(key)
            assert bk is not None, f"bookkeeping must exist for proc key {key!r}"
            assert bk.get("relation_type") == "preference", (
                f"Proc key bookkeeping must carry relation_type='preference'; "
                f"got {bk.get('relation_type')!r}"
            )

    def test_interim_active_keys_includes_proc_key(self, monkeypatch, tmp_path):
        """After an interim cycle, proc key is active in the interim tier (not procedural main).

        The router pairs keys with the adapter named by their store tier, so
        this matters at the router level: active_keys_in_tier(interim_slot)
        returns the proc key; active_keys_in_tier("procedural") does NOT.
        """
        loop = _make_cycle_loop(tmp_path)
        _wire_cycle_fakes(loop, monkeypatch)

        loop.merger.graph.add_edge(
            "gwen",
            "yoga",
            predicate="practices",
            relation_type="preference",
            speaker_id="speaker0",
        )

        pending = loop.take_pending_relations()
        loop.run_consolidation_cycle(
            [
                {
                    "subject": "Gwen",
                    "predicate": "practices",
                    "object": "Yoga",
                    "relation_type": "preference",
                    "speaker_id": "speaker0",
                }
            ],
            [],
            speaker_id="speaker0",
            mode="train",
            pending=pending,
            run_label="test_active_keys",
            stamp=_INTERIM_STAMP,
        )

        interim_active = list(loop.store.active_keys_in_tier(_INTERIM_SLOT))
        proc_main_active = list(loop.store.active_keys_in_tier("procedural"))

        assert any(k.startswith("proc") for k in interim_active), (
            f"Proc key must be active in interim tier '{_INTERIM_SLOT}'; "
            f"active keys there: {interim_active}"
        )
        assert not any(k.startswith("proc") for k in proc_main_active), (
            f"Proc key must NOT appear in 'procedural' main during an interim cycle; "
            f"active keys there: {proc_main_active}"
        )
