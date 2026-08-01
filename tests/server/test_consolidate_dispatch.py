"""Tests for the server-side consolidation dispatch infrastructure.

All tests are CPU-only, no model load required.  The consolidation-loop and
BackgroundTrainer are mocked so the implementation-level dispatch paths can be
verified in isolation.

Coverage:
- ``_consolidation_dispatch_guards`` shared guard helper
- ``_dispatch_consolidation`` — the arbitrator: ``AUTO`` is requested ONLY by
  ``/scheduled-tick`` (deadline resolution via ``_is_full_cycle_due``, the
  catch-up gate, and the cadence stamp are its business alone).  ``FULL`` and
  ``INTERIM`` are each requestable directly (``/consolidate``,
  ``/consolidate/interim``) as well as via ``AUTO``'s resolution, and the
  content gate applies identically either way — a manual door drops only the
  TIME condition, never the CONTENT condition.  ``RECONCILE`` is the one
  action exempt from the content gate (still subject to the shared safety
  guards).  Also covers the executor submission ritual and the concurrency
  guard.
- ``_run_full_consolidation_sync`` noop terminal: an empty ``tiers_rebuilt``
  ends the cycle as a noop, and the sessions consumed by the pre-stage are
  still retired so they cannot accumulate unboundedly.

The fold itself carries no caller intent: the arbitrator decides whether a
dispatch has anything to consolidate, and
``loop.consolidate(mode=..., keys_from=...)`` then does what it is told with
the venue and key source it was handed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_router() -> MagicMock:
    """Build a minimal mock router that supports reload()."""
    r = MagicMock()
    r.reload.return_value = None
    return r


def _make_dispatch_state(
    *,
    mode: str = "local",
    consolidating: bool = False,
    bg_is_training: bool = False,
    consolidation_mode: str = "train",
    max_interim_count: int = 7,
    tmp_path=None,
) -> dict:
    """Minimal ``_state`` dict for consolidation-dispatch tests.

    Args:
        mode: Runtime mode ("local" or "cloud-only").
        consolidating: Whether ``_state["consolidating"]`` is already True.
        bg_is_training: Whether the BackgroundTrainer reports active training.
        consolidation_mode: Value for ``config.consolidation.mode``.
        max_interim_count: Value for ``config.consolidation.max_interim_count``.
        tmp_path: When provided, set ``config.paths.data`` to this real path so
            that incident/run-status I/O writes land in ``tmp/state/`` rather
            than creating a literal ``MagicMock/`` directory at the repo root.
            Tests that exercise the full cycle path (``_run_full_consolidation_sync``)
            must supply this; tests that only exercise dispatch guards do not.
    """
    mock_config = MagicMock()
    mock_config.consolidation.mode = consolidation_mode
    mock_config.consolidation.max_interim_count = max_interim_count
    # Prevent ThermalPolicy.from_consolidation_config from comparing a MagicMock.
    mock_config.consolidation.training_temp_limit = 0
    # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown fold gate.
    mock_config.vram.cooldown_gate_threshold_c = 0
    # Ground incident/run-status I/O in a real path so the writes land in the
    # pytest tmp directory instead of creating a MagicMock/ tree at repo root.
    if tmp_path is not None:
        mock_config.paths.data = tmp_path

    mock_loop = MagicMock()
    mock_loop.model = MagicMock(name="model")
    mock_loop.shutdown_requested = False
    mock_loop.store.replay_enabled = False
    # Default fold return: successful noop-ish result with tiers_rebuilt=[].
    mock_loop.consolidate.return_value = {
        "tiers_rebuilt": [],
        "graph_drift_count": 0,
        "drift_deduplicated": 0,
        "drift_orphan": 0,
        "drift_genuine_loss": 0,
        "keys_per_tier": {},
        "rolled_back": False,
        "rollback_tier": None,
        "tier_delta": {},
    }

    bg = None
    if bg_is_training:
        bg = MagicMock()
        bg.is_training = True

    return {
        "config": mock_config,
        "model": MagicMock(name="model"),
        "tokenizer": MagicMock(name="tokenizer"),
        "consolidation_loop": mock_loop,
        "session_buffer": MagicMock(),
        "router": _make_mock_router(),
        "background_trainer": bg,
        "consolidating": consolidating,
        "mode": mode,
        "cloud_only_reason": None,
        "last_consolidation": None,
        "last_consolidation_result": None,
        "last_consolidation_error": None,
        "event_loop": None,
        "migration": {},
    }


# ---------------------------------------------------------------------------
# TestConsolidationDispatchGuards
# ---------------------------------------------------------------------------


class TestConsolidationDispatchGuards:
    """_consolidation_dispatch_guards returns the right block reason or None."""

    def test_returns_none_when_clear(self, monkeypatch) -> None:
        """All guards pass → returns None (proceed)."""
        import paramem.server.app as app_module

        state = _make_dispatch_state()
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() is None

    def test_deferred_already_running(self, monkeypatch) -> None:
        """consolidating=True → deferred_already_running."""
        import paramem.server.app as app_module

        state = _make_dispatch_state(consolidating=True)
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_already_running"

    def test_deferred_cloud_only(self, monkeypatch) -> None:
        """mode=cloud-only → deferred_cloud_only."""
        import paramem.server.app as app_module

        state = _make_dispatch_state(mode="cloud-only")
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_cloud_only"

    def test_deferred_bg_training(self, monkeypatch) -> None:
        """BackgroundTrainer.is_training=True → deferred_bg_training."""
        import paramem.server.app as app_module

        state = _make_dispatch_state(bg_is_training=True)
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_bg_training"

    def test_deferred_trial_active(self, monkeypatch) -> None:
        """migration.state == 'TRIAL' → deferred_trial_active.

        The same predicate (``_trial_active``) that makes ``require_no_trial``
        409 the REST routes — mirrored here so an in-process caller (never
        resolving FastAPI dependencies) is refused too.
        """
        import paramem.server.app as app_module

        state = _make_dispatch_state()
        state["migration"] = {"state": "TRIAL"}
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_trial_active"


# ---------------------------------------------------------------------------
# TestConsolidationArbitrator — action resolution + the ONE content gate
# ---------------------------------------------------------------------------


def _make_arbitrator_state(
    tmp_path,
    *,
    consolidation_mode: str = "train",
    max_interim_count: int = 7,
    named_sessions: int = 0,
    anon_sessions: int = 0,
    refresh_cadence: str = "12h",
    period_seconds: "int | None" = None,
) -> dict:
    """``_state`` for arbitrator tests, with a REAL SessionBuffer and adapter dir.

    The content gate reads both — the on-disk interim set (through the
    payload-aware ``iter_interim_dirs``) and the pending-session buffer — so
    neither may be a MagicMock: a mock would satisfy the gate by accident.

    Args:
        tmp_path: pytest tmp dir; becomes ``config.paths.data`` and the parent
            of ``config.adapter_dir``.
        consolidation_mode: ``config.consolidation.mode`` (the payload venue).
        max_interim_count: ``config.consolidation.max_interim_count`` (N).
        named_sessions: Number of pending NAMED sessions to seed.
        anon_sessions: Number of pending UNIDENTIFIABLE sessions to seed (no
            speaker id, no voice embedding).
        refresh_cadence: ``config.consolidation.refresh_cadence``.  For any
            real cadence (default ``"12h"``) this fixture pre-seeds a durable
            last-scheduled-run stamp well before the current mark, so an
            ``AUTO`` tick reads DUE and reaches the arbitrator's own gates
            instead of seed-and-noop on a virgin stamp file (the universal
            catch-up gate — ``schedule_grammar.scheduled_run_due`` — applies
            to every cadence kind, not only non-calendar-exact ones).  Tests
            that specifically exercise catch-up-gate semantics build their
            own stamp state directly (see ``TestSchedulerCatchUpGate`` in
            ``tests/test_consolidation.py``).  ``""`` is manual-only (no
            cadence, no stamp seeded) — used by the manual-only-posture
            tests, where ``FULL``/``INTERIM`` requested directly are the
            only doors that ever fire.
        period_seconds: ``config.consolidation.consolidation_period_seconds`` —
            the full-fold period ``_is_full_cycle_due`` measures against, read
            ONLY by the ``AUTO`` (scheduled-tick) path.  ``None`` (the
            default) is a manual-only cadence: no deadline, so
            ``_is_full_cycle_due`` is False for any interim ring.  A directly
            requested ``FULL`` never reads this at all.
    """
    from paramem.server.schedule_grammar import parse_schedule_atom
    from paramem.server.schedule_state import write_last_scheduled_run
    from paramem.server.session_buffer import SessionBuffer

    cfg = MagicMock()
    cfg.consolidation.mode = consolidation_mode
    cfg.consolidation.max_interim_count = max_interim_count
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.consolidation.training_idle_debounce_s = 30
    cfg.consolidation.orphan_retirement_seconds = None
    cfg.consolidation.retain_sessions = False
    cfg.consolidation.consolidation_period_seconds = period_seconds
    cfg.debug = False
    cfg.paths.data = tmp_path
    cfg.adapter_dir = tmp_path / "adapters"
    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)

    # Universal catch-up gate: pre-seed a DUE stamp (well before the current
    # mark) for any real cadence so an AUTO tick reaches the arbitrator
    # instead of seed-and-noop on a virgin stamp file — see the
    # refresh_cadence docstring above.
    _atom = parse_schedule_atom(refresh_cadence)
    if _atom is not None and _atom.kind != "off":
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)

    buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
    for i in range(named_sessions):
        buffer.append(f"conv-named-{i}", "user", "Hello", speaker_id=f"speaker{i + 1}")
        buffer.append(f"conv-named-{i}", "assistant", "Hi")
    for i in range(anon_sessions):
        buffer.append(f"conv-anon-{i}", "user", "Hello")
        buffer.append(f"conv-anon-{i}", "assistant", "Hi")

    store = MagicMock()
    store.is_anonymous.return_value = False

    return {
        "config": cfg,
        "session_buffer": buffer,
        "speaker_store": store,
        "consolidating": False,
        "mode": "local",
        "background_trainer": None,
        "cloud_only_reason": None,
        "last_chat_monotonic": None,
        "pending_rehydration": False,
        "store_load_degraded": False,
    }


def _make_interim_slot(adapter_dir, stamp: str, *, payload: str | None) -> None:
    """Create ``episodic/interim_<stamp>/`` with (or without) a venue payload.

    Args:
        adapter_dir: Adapter root.
        stamp: ``YYYYMMDDTHHMM`` interim stamp.
        payload: ``"graph"`` → ``graph.json`` (simulate venue); ``"weights"`` →
            ``adapter_model.safetensors`` (train venue); ``None`` → a
            payload-less shell (the torn-write case the gate must ignore).
    """
    d = adapter_dir / "episodic" / f"interim_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    if payload == "graph":
        (d / "graph.json").write_text("{}")
    elif payload == "weights":
        slot = d / f"{stamp}-slot"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")


class _ExecutorSpy:
    """Stand-in for the event loop: records what was submitted, runs nothing."""

    def __init__(self) -> None:
        self.submitted: list[object] = []
        self.loop = MagicMock()
        self.loop.run_in_executor.side_effect = self._submit

    def _submit(self, executor, fn):
        self.submitted.append(fn)
        future = MagicMock()
        future.add_done_callback.return_value = None
        return future

    @property
    def call_count(self) -> int:
        return len(self.submitted)


def _dispatch(state, action, *, monkeypatch=None):
    """Run the arbitrator against *state*, capturing executor submissions.

    Returns ``(status, resolved_action, spy, due_calls)`` where ``due_calls``
    counts the ``_is_full_cycle_due`` invocations.  ``AUTO`` is the only
    action that ever calls it — a directly requested ``FULL``/``INTERIM``
    never does, so ``due_calls == []`` is itself a load-bearing assertion for
    every manual-door test in this module.
    """
    import paramem.server.app as app_module

    spy = _ExecutorSpy()
    due_calls: list[bool] = []
    _real_due = app_module._is_full_cycle_due

    def _counting_due(config):
        result = _real_due(config)
        due_calls.append(result)
        return result

    state["event_loop"] = spy.loop
    monkeypatch.setattr(app_module, "_state", state)
    monkeypatch.setattr(app_module, "_is_full_cycle_due", _counting_due)
    monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
    status, resolved = app_module._dispatch_consolidation(action)
    return status, resolved, spy, due_calls


def _submitted_full_fold_key_sources(spy) -> list[str]:
    """The ``keys_from`` each submitted full fold was bound to.

    The arbitrator submits ``functools.partial(_run_full_consolidation_sync,
    keys_from)``, so the key source it chose for the fold is readable off the
    partial's bound arguments — which is what distinguishes a FULL dispatch
    from a RECONCILE one below the arbitrator.
    """
    import paramem.server.app as app_module

    sources = []
    for fn in spy.submitted:
        assert getattr(fn, "func", None) is app_module._run_full_consolidation_sync, (
            f"not a full-fold submission: {fn!r}"
        )
        sources.append(fn.args[0])
    return sources


class TestConsolidationArbitrator:
    """_dispatch_consolidation: action resolution, the content gate, dispatch."""

    def test_count_zero_no_pending_is_a_noop_and_submits_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """N=0, scheduled tick, ZERO pending sessions → noop, no executor submission.

        The regression test for the defect this gate exists to fix: at
        max_interim_count==0 ``_is_full_cycle_due`` is unconditionally True, so
        every scheduled tick used to retrain every main tier with nothing new to
        learn.  Asserting the status alone is not enough — the load-bearing
        assertion is that NOTHING was submitted to the executor.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "noop_no_pending"
        assert resolved is ConsolidationAction.FULL
        assert spy.call_count == 0, "a full GPU retrain was dispatched with nothing to learn"
        assert state["consolidating"] is False
        assert due_calls == [True], "the schedule gate must be consulted exactly once"

    def test_count_zero_unattributable_sessions_are_retired_on_the_full_path(
        self, tmp_path, monkeypatch
    ) -> None:
        """N=0, scheduled tick, only UNIDENTIFIABLE sessions pending → noop_no_named,
        no executor submission, AND the sessions are retired.

        At max_interim_count==0 the interim path never runs, so nothing else
        retires UNIDENTIFIABLE/expired-HOLDABLE sessions; before the triage
        pre-stage existed they accumulated in the buffer forever.  This is the
        regression test for that leak.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, anon_sessions=2)
        assert len(state["session_buffer"].pending_facts()) == 2, (
            "fixture sanity: two pending sessions before dispatch"
        )

        status, resolved, spy, _due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "noop_no_named"
        assert resolved is ConsolidationAction.FULL
        assert spy.call_count == 0, "a full GPU retrain was dispatched with nothing to learn"
        assert state["session_buffer"].pending_facts() == [], (
            "UNIDENTIFIABLE sessions must be retired on the full path at N=0, "
            "not left to accumulate forever"
        )

    def test_count_zero_with_pending_dispatches_the_full_fold(self, tmp_path, monkeypatch) -> None:
        """N=0 + NAMED pending sessions → the full fold runs (and will consume them)."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_key_sources(spy) == ["all_tiers"]
        assert state["consolidating"] is True

    def test_full_due_with_content_bearing_interims_dispatches(self, tmp_path, monkeypatch) -> None:
        """N>0, oldest interim past the full period, content-bearing → dispatches."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=2, period_seconds=1)
        adapter_dir = state["config"].adapter_dir
        # Aged stamps + a 1-second full period → the deadline has long passed.
        for i in range(3):
            _make_interim_slot(adapter_dir, f"2020010{i + 1}T0000", payload="weights")

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_key_sources(spy) == ["all_tiers"]
        assert due_calls == [True]

    def test_full_door_noops_with_no_interims_and_no_pending(self, tmp_path, monkeypatch) -> None:
        """An explicitly requested FULL (``/consolidate``) on an empty store noops.

        No deadline check at all — ``_is_full_cycle_due`` is never consulted
        (``due_calls == []``) — only the content gate: no content-bearing
        interim slot, and at ``max_interim_count > 0`` a pending session
        (there is none here either) would not be this fold's content anyway.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "noop_no_interim_slots"
        assert resolved is ConsolidationAction.FULL
        assert spy.call_count == 0
        assert due_calls == [], "a directly requested FULL must never consult the deadline gate"

    def test_full_door_noops_with_no_interim_slots_even_with_a_pending_session(
        self, tmp_path, monkeypatch
    ) -> None:
        """FULL, zero payload-bearing interims, N>0 → noop even with a pending session.

        At max_interim_count > 0 the FULL fold never consumes pending sessions
        directly — that is the INTERIM tier's job — so a pending NAMED session
        is not input to THIS fold and must not let it proceed.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "noop_no_interim_slots"
        assert resolved is ConsolidationAction.FULL
        assert spy.call_count == 0
        assert due_calls == []

    def test_full_door_dispatches_with_a_payload_bearing_interim_slot(
        self, tmp_path, monkeypatch
    ) -> None:
        """FULL, one content-bearing interim slot, N>0 → dispatches.

        No deadline math involved: the slot alone is enough, whatever
        ``_is_full_cycle_due`` would have said.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_key_sources(spy) == ["all_tiers"]
        assert due_calls == []

    def test_full_door_at_count_zero_dispatches_on_pending_named_session_alone(
        self, tmp_path, monkeypatch
    ) -> None:
        """FULL at max_interim_count==0, no interim slots, one NAMED session → dispatches.

        At this count no interim tier exists at all, so the fold's own
        content is the pending session it will consume directly
        (``consume_pending=True`` inside ``_run_full_consolidation_sync``).
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_key_sources(spy) == ["all_tiers"]
        assert due_calls == []

    def test_full_door_at_count_zero_absorbs_a_leftover_interim_slot(
        self, tmp_path, monkeypatch
    ) -> None:
        """FULL at max_interim_count==0 with a leftover payload-bearing interim slot dispatches.

        Simulates an operator lowering ``max_interim_count`` from >0 to 0
        after a slot was already minted: the slot is still on disk, still
        payload-bearing, and must not be stranded.  The interim-slot check
        runs unconditionally (not gated on the CURRENT count), so it is
        absorbed and reaped via ``keys_from="all_tiers"`` even though no
        pending session exists at all.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0)
        _make_interim_slot(state["config"].adapter_dir, "20260101T0000", payload="weights")

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_key_sources(spy) == ["all_tiers"]
        assert due_calls == []

    def test_reconcile_dispatches_the_fold_over_the_main_tiers_only(
        self, tmp_path, monkeypatch
    ) -> None:
        """RECONCILE reaches the fold with the narrowed key source.

        This is the whole difference between the two full-fold doors below the
        arbitrator: same entry point, same status, ``keys_from="main_tiers"``.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE
        assert _submitted_full_fold_key_sources(spy) == ["main_tiers"]
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit RECONCILE"

    def test_reconcile_dispatches_on_an_empty_store_too(self, tmp_path, monkeypatch) -> None:
        """RECONCILE never reaches the content gate — an empty store still dispatches."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE
        assert _submitted_full_fold_key_sources(spy) == ["main_tiers"]
        assert due_calls == []

    def test_fold_entry_takes_venue_key_source_and_fold_inputs_only(self, tmp_path) -> None:
        """The arbitrator's intent stays in the arbitrator.

        The fold entry's parameter set is pinned by exact set equality, so any
        caller-intent parameter leaking down from the dispatch layer fails here.
        ``keys_from`` is not caller intent: it names the fold's key source, and
        two different doors map onto the same two values.
        """
        import inspect

        from paramem.training.consolidation import ConsolidationLoop

        params = inspect.signature(ConsolidationLoop.consolidate).parameters
        assert set(params) == {
            "self",
            "mode",
            "keys_from",
            "consume_pending",
            "trainer",
            "router",
        }, f"unexpected fold-entry parameters: {sorted(params)}"

    def test_interim_door_noops_with_no_pending_sessions(self, tmp_path, monkeypatch) -> None:
        """An explicitly requested INTERIM with zero pending sessions noops.

        The content gate applies to a direct INTERIM request exactly as it
        applies to the schedule's own interim resolution: with nothing
        pending there is nothing to extract or train.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "noop_no_pending"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.call_count == 0
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit INTERIM"

    def test_unattributable_sessions_are_retired_on_every_door(self, tmp_path, monkeypatch) -> None:
        """Orphan retirement is a pre-stage, not part of the content gate.

        ``RECONCILE`` bypasses the content gate entirely and the other three
        still noop here (nothing NAMED pending) — retirement must not depend
        on either.  Pinned on each of the four production doors in turn:
        ``/scheduled-tick`` (AUTO), ``/consolidate`` (FULL),
        ``/consolidate/interim`` (INTERIM), ``/reconsolidate`` (RECONCILE).
        """
        from paramem.server.app import ConsolidationAction

        for action in (
            ConsolidationAction.AUTO,
            ConsolidationAction.FULL,
            ConsolidationAction.INTERIM,
            ConsolidationAction.RECONCILE,
        ):
            state = _make_arbitrator_state(tmp_path / action.value, anon_sessions=2)
            assert len(state["session_buffer"].pending_facts()) == 2, "fixture sanity"

            _dispatch(state, action, monkeypatch=monkeypatch)

            assert state["session_buffer"].pending_facts() == [], (
                f"{action.value}: UNIDENTIFIABLE sessions must be retired on every dispatch"
            )

    def test_interim_with_named_sessions_dispatches_the_interim_path(
        self, tmp_path, monkeypatch
    ) -> None:
        """NAMED pending sessions → the interim extract+train path is submitted."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "started"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._extract_and_start_training]

    def test_interim_at_count_zero_is_refused(self, tmp_path, monkeypatch) -> None:
        """An explicit INTERIM at max_interim_count==0 → noop_no_interim_tier.

        There is no interim tier at N=0; the request is meaningless.  This is a
        tier check, not the content gate: NAMED sessions are pending here.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        status, _resolved, spy, _ = _dispatch(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "noop_no_interim_tier"
        assert spy.call_count == 0

    @pytest.mark.parametrize("consolidation_mode", ["train", "simulate"])
    def test_simulate_and_train_reach_the_same_dispatch_decision(
        self, tmp_path, monkeypatch, consolidation_mode
    ) -> None:
        """Identical inputs → identical dispatch decision in both venues.

        The content gate reads each venue's own payload, so the SAME logical
        input (three content-bearing interim slots, aged past the full period)
        must produce the same outcome in simulate and in train.
        """
        from paramem.server.app import ConsolidationAction

        payload = "graph" if consolidation_mode == "simulate" else "weights"
        state = _make_arbitrator_state(
            tmp_path,
            consolidation_mode=consolidation_mode,
            max_interim_count=2,
            period_seconds=1,
        )
        for i in range(3):
            _make_interim_slot(state["config"].adapter_dir, f"2020010{i + 1}T0000", payload=payload)

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert (status, resolved, spy.call_count) == (
            "started_full",
            ConsolidationAction.FULL,
            1,
        )
        assert due_calls == [True]

    @pytest.mark.parametrize("consolidation_mode", ["train", "simulate"])
    def test_simulate_and_train_noop_identically_on_empty_input(
        self, tmp_path, monkeypatch, consolidation_mode
    ) -> None:
        """Nothing on disk, nothing pending → the same noop in both venues."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(
            tmp_path, consolidation_mode=consolidation_mode, max_interim_count=2
        )
        status, _resolved, spy, _ = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "noop_no_pending"
        assert spy.call_count == 0

    def test_second_dispatch_is_serialized_by_the_consolidating_guard(
        self, tmp_path, monkeypatch
    ) -> None:
        """The first dispatch sets ``consolidating``; the second defers.

        ``_dispatch_to_executor`` sets the flag on the event-loop thread BEFORE
        submitting, so there is no window in which a second dispatch can slip a
        concurrent fold past the guard.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        first, _a1, spy1, _ = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)
        second, _a2, spy2, _ = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert first == "started_full"
        assert spy1.call_count == 1
        assert second == "deferred_already_running"
        assert spy2.call_count == 0, "a second fold must never be submitted concurrently"

    def test_idle_debounce_applies_to_a_manual_full_request(self, tmp_path, monkeypatch) -> None:
        """The debounce is a safety property, not a schedule — it defers every action.

        A chat turn inside the debounce window defers even an explicitly
        requested full fold: the fold would seize the GPU from a live
        conversation.
        """
        import time as _time

        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["last_chat_monotonic"] = _time.monotonic() - 5  # debounce is 30 s

        status, _resolved, spy, _ = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "deferred_idle"
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# TestStampPredicate — the schedule stamp belongs to the schedule.  A manual
# run does not move the cadence window; the next scheduled tick still has its
# content gate and noops on its own if the manual run consumed everything.
# ---------------------------------------------------------------------------


class TestStampPredicate:
    """``_stamp_scheduled_run`` fires iff the dispatch resolved from ``AUTO``."""

    def _dispatch_and_track_stamp(self, state, action, *, monkeypatch) -> "tuple[str, object, int]":
        """Run the arbitrator, counting real (unmocked) ``_stamp_scheduled_run`` calls.

        The real function still runs (so the persisted stamp can be checked
        separately) — this only counts invocations.
        """
        import paramem.server.app as app_module

        spy = _ExecutorSpy()
        state["event_loop"] = spy.loop
        stamp_calls: list[object] = []
        _real_stamp = app_module._stamp_scheduled_run

        def _counting_stamp(config):
            stamp_calls.append(config)
            return _real_stamp(config)

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_stamp_scheduled_run", _counting_stamp)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        status, resolved = app_module._dispatch_consolidation(action)
        return status, resolved, len(stamp_calls)

    @pytest.mark.parametrize("action_name", ["FULL", "INTERIM", "RECONCILE"])
    def test_no_manual_door_moves_the_cadence_window(
        self, tmp_path, monkeypatch, action_name
    ) -> None:
        """Every DIRECTLY REQUESTED action dispatches WITHOUT stamping.

        Checked two ways — the wrapped ``_stamp_scheduled_run`` is never called,
        AND (using a non-calendar-exact cadence, where the stamp is a real
        on-disk write) the persisted ``last_scheduled_run`` is byte-identical
        before and after.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(
            tmp_path,
            max_interim_count=2,
            named_sessions=1,
            refresh_cadence="every 5h",
        )
        for i in range(2):
            _make_interim_slot(
                state["config"].adapter_dir, f"2026070{i + 1}T0000", payload="weights"
            )
        state_dir = state["config"].paths.data / "state"
        seeded_stamp = time.time() - 6 * 3600
        write_last_scheduled_run(state_dir, seeded_stamp)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, getattr(ConsolidationAction, action_name), monkeypatch=monkeypatch
        )

        assert status in {"started", "started_full"}
        assert resolved is getattr(ConsolidationAction, action_name)
        assert stamp_calls == 0, "a manual run must not reset the cadence window"
        assert read_last_scheduled_run(state_dir) == seeded_stamp

    @pytest.mark.parametrize("action_name", ["FULL", "INTERIM"])
    def test_no_manual_door_moves_the_cadence_window_on_a_noop(
        self, tmp_path, monkeypatch, action_name
    ) -> None:
        """A directly requested FULL/INTERIM that the content gate noops still
        does not stamp.

        The stamp belongs to the schedule regardless of the door's outcome —
        a manual noop must not consume the next scheduled tick's own content
        gate either.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(
            tmp_path,
            max_interim_count=2,
            refresh_cadence="every 5h",
        )
        state_dir = state["config"].paths.data / "state"
        seeded_stamp = time.time() - 6 * 3600
        write_last_scheduled_run(state_dir, seeded_stamp)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, getattr(ConsolidationAction, action_name), monkeypatch=monkeypatch
        )

        assert status.startswith("noop_")
        assert resolved is getattr(ConsolidationAction, action_name)
        assert stamp_calls == 0, "a manual noop must not reset the cadence window"
        assert read_last_scheduled_run(state_dir) == seeded_stamp

    def test_scheduled_full_stamps(self, tmp_path, monkeypatch) -> None:
        """AUTO resolving to FULL stamps: it IS the scheduled cycle."""
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=2, refresh_cadence="every 5h", period_seconds=1
        )
        for i in range(3):
            _make_interim_slot(
                state["config"].adapter_dir, f"2020010{i + 1}T0000", payload="weights"
            )
        # Seed the catch-up stamp far enough back that this tick is due.
        from paramem.server.schedule_state import write_last_scheduled_run

        write_last_scheduled_run(state["config"].paths.data / "state", time.time() - 6 * 3600)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert stamp_calls == 1
        assert read_last_scheduled_run(state["config"].paths.data / "state") is not None

    def test_manual_full_does_not_stamp_even_with_a_deadline_that_has_passed(
        self, tmp_path, monkeypatch
    ) -> None:
        """A directly requested FULL never stamps, even on content identical to
        :meth:`test_scheduled_full_stamps` where the deadline has passed.

        The only difference between the two is which action was requested —
        proving the stamp is keyed on ``AUTO``, not on the fold's outcome.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=2, refresh_cadence="every 5h", period_seconds=1
        )
        for i in range(3):
            _make_interim_slot(
                state["config"].adapter_dir, f"2020010{i + 1}T0000", payload="weights"
            )
        seeded_stamp = time.time() - 6 * 3600
        write_last_scheduled_run(state["config"].paths.data / "state", seeded_stamp)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert stamp_calls == 0
        assert read_last_scheduled_run(state["config"].paths.data / "state") == seeded_stamp, (
            "a directly requested FULL must not move the cadence window"
        )

    def test_scheduled_interim_stamps(self, tmp_path, monkeypatch) -> None:
        """AUTO resolving to INTERIM stamps for the same reason."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, named_sessions=1, refresh_cadence="every 5h"
        )
        from paramem.server.schedule_state import write_last_scheduled_run

        write_last_scheduled_run(state["config"].paths.data / "state", time.time() - 6 * 3600)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started"
        assert resolved is ConsolidationAction.INTERIM
        assert stamp_calls == 1


# ---------------------------------------------------------------------------
# TestUniversalCatchUpGate — the durable-stamp catch-up gate
# (schedule_grammar.scheduled_run_due) applies to EVERY cadence kind, not
# only non-calendar-exact ("heartbeat") ones. TestSchedulerCatchUpGate in
# tests/test_consolidation.py covers the same contract for non-exact
# cadences ("every 5h"); this class exercises it for a calendar-exact
# cadence ("12h", the server.yaml default) to prove the gate is now
# universal rather than a heartbeat-only special case.
# ---------------------------------------------------------------------------


class TestUniversalCatchUpGate:
    def _virgin_stamp_state(self, tmp_path, *, refresh_cadence: str, **kwargs) -> dict:
        """An arbitrator state for *refresh_cadence* with NO durable stamp on disk.

        Constructed with ``refresh_cadence=""`` first so ``_make_arbitrator_state``'s
        own auto-seed (see its docstring) never fires, then the cadence is set
        to the real value the test wants to exercise — giving a virgin stamp
        file under a real (non-off) cadence without touching the shared
        fixture's default behaviour for every other test in this module.
        """
        state = _make_arbitrator_state(tmp_path, refresh_cadence="", **kwargs)
        state["config"].consolidation.refresh_cadence = refresh_cadence
        return state

    def test_first_auto_tick_on_an_exact_cadence_seeds_without_dispatching(
        self, tmp_path, monkeypatch
    ) -> None:
        """Calendar-exact '12h' with no stamp on disk → noop_scheduler_seeded,
        the stamp file is created, and nothing is submitted — the same
        seed-and-noop contract non-exact cadences have always had, now
        applying to an exact cadence too.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run

        state = self._virgin_stamp_state(tmp_path, refresh_cadence="12h", max_interim_count=7)
        state_dir = state["config"].paths.data / "state"
        assert read_last_scheduled_run(state_dir) is None

        status, _resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "noop_scheduler_seeded"
        assert read_last_scheduled_run(state_dir) is not None
        assert spy.call_count == 0
        assert due_calls == [], "_is_full_cycle_due must not be reached on a virgin stamp"

    def test_second_tick_inside_the_same_mark_window_is_not_due(
        self, tmp_path, monkeypatch
    ) -> None:
        """A stamp written at the current 12h mark reads NOT_DUE for a second
        tick still inside that mark's window."""
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_grammar import scheduled_run_stamp_value
        from paramem.server.schedule_state import write_last_scheduled_run

        state = self._virgin_stamp_state(tmp_path, refresh_cadence="12h", max_interim_count=7)
        state_dir = state["config"].paths.data / "state"
        write_last_scheduled_run(state_dir, scheduled_run_stamp_value("12h", time.time()))

        status, _resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "noop_not_due"
        assert spy.call_count == 0
        assert due_calls == []

    def test_tick_after_a_mark_crossing_reaches_the_arbitrator(self, tmp_path, monkeypatch) -> None:
        """A stamp from a previous 12h mark reads DUE and reaches the content
        gate (a real ``noop_no_pending`` outcome, not a seed/not-due
        short-circuit) — proven by ``_is_full_cycle_due`` actually being
        consulted.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import write_last_scheduled_run

        state = self._virgin_stamp_state(tmp_path, refresh_cadence="12h", max_interim_count=7)
        state_dir = state["config"].paths.data / "state"
        write_last_scheduled_run(state_dir, time.time() - 86400)  # a full day back

        status, _resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "noop_no_pending"
        assert due_calls == [False]
        assert spy.call_count == 0

    def test_deferred_tick_does_not_advance_the_stamp(self, tmp_path, monkeypatch) -> None:
        """A tick blocked by ``_consolidation_dispatch_guards`` (already
        running) must not stamp — the next tick is still DUE."""
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = self._virgin_stamp_state(tmp_path, refresh_cadence="12h", max_interim_count=7)
        state_dir = state["config"].paths.data / "state"
        old_stamp = time.time() - 86400
        write_last_scheduled_run(state_dir, old_stamp)
        state["consolidating"] = True  # -> _consolidation_dispatch_guards() blocks

        status, _resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "deferred_already_running"
        assert read_last_scheduled_run(state_dir) == old_stamp, (
            "a deferred tick must not consume the cadence window"
        )
        assert due_calls == []
        assert spy.call_count == 0

    def test_calendar_exact_12h_tick_exactly_at_the_period_boundary_is_due(
        self, tmp_path, monkeypatch
    ) -> None:
        """A stamp at the 00:00 mark, evaluated exactly at the following
        12:00:00 mark, must read DUE via mark-crossing — not via a
        period-elapsed-seconds comparison that a few seconds of dispatch
        delay around the boundary could throw off into a false 'not due'.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_grammar import previous_mark
        from paramem.server.schedule_state import write_last_scheduled_run

        state = self._virgin_stamp_state(tmp_path, refresh_cadence="12h", max_interim_count=7)
        state_dir = state["config"].paths.data / "state"

        midnight_mark = previous_mark("12h", time.time())
        write_last_scheduled_run(state_dir, midnight_mark)
        tick_time = midnight_mark + 12 * 3600  # the very next 12h mark, to the second

        with patch("paramem.server.schedule_grammar.time.time", return_value=tick_time):
            status, _resolved, spy, due_calls = _dispatch(
                state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
            )

        assert status == "noop_no_pending", (
            "the tick must reach the content gate (DUE) exactly at the mark boundary"
        )
        assert due_calls == [False]
        assert spy.call_count == 0

    def test_trial_active_defers_in_process_but_rest_route_still_409s(
        self, tmp_path, monkeypatch
    ) -> None:
        """A migration TRIAL: the in-process arbitrator defers
        (``deferred_trial_active``); the REST route never reaches the
        arbitrator at all — ``require_no_trial`` 409s first.  Same predicate
        (``_trial_active``), two consumers.
        """
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["migration"] = {"state": "TRIAL"}
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)

        result, _action = app_module._dispatch_consolidation(ConsolidationAction.AUTO)
        assert result == "deferred_trial_active"

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/scheduled-tick")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"


# ---------------------------------------------------------------------------
# TestConsolidationRoutes — the operator surface: four intent-named, body-less
# doors onto the one arbitrator.  ``/consolidate`` requests ``FULL`` and
# ``/consolidate/interim`` requests ``INTERIM`` directly — the identical
# content check the schedule's own resolution would apply, minus the deadline
# math.  ``/scheduled-tick`` is the only door that requests ``AUTO``.  None of
# them exposes an internal knob (no mode, no force, no request body at all).
# ---------------------------------------------------------------------------


def _route_client(state, monkeypatch) -> "tuple[object, list[tuple[object, str]]]":
    """TestClient over the app with *state* installed and the executor stubbed.

    The real ``_dispatch_to_executor`` is replaced by a recorder: the arbitrator
    (guards, schedule resolution, content gate) runs for real, but nothing is
    submitted to a thread pool.

    Returns:
        ``(client, submitted)`` — ``submitted`` collects ``(fn, status)`` for
        every dispatch that reached the executor ritual.
    """
    from fastapi.testclient import TestClient

    import paramem.server.app as app_module

    submitted: list[tuple[object, str]] = []

    def _record(fn, status):
        submitted.append((fn, status))
        return status

    state.setdefault("migration", {})
    monkeypatch.setattr(app_module, "_state", state)
    monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
    monkeypatch.setattr(app_module, "_dispatch_to_executor", _record)
    return TestClient(app_module.app, raise_server_exceptions=False), submitted


def _route_key_sources(submitted) -> list[str]:
    """The ``keys_from`` bound to each full fold that reached the executor ritual."""
    import paramem.server.app as app_module

    sources = []
    for fn, _status in submitted:
        assert getattr(fn, "func", None) is app_module._run_full_consolidation_sync, (
            f"not a full-fold submission: {fn!r}"
        )
        sources.append(fn.args[0])
    return sources


class TestConsolidationRoutes:
    """The four consolidation routes: intent → arbitrator call → status/action."""

    def test_consolidate_collapses_the_interims_regardless_of_the_schedule(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/consolidate`` folds a content-bearing interim slot even though the
        schedule would not (yet) call a full cycle due.

        One content-bearing interim slot at N=7 with no deadline configured:
        ``_is_full_cycle_due`` would be False for a scheduled tick, but
        ``/consolidate`` requests ``FULL`` directly and never consults it —
        content alone decides.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "started_full", "action": "full"}
        assert _route_key_sources(submitted) == ["all_tiers"]

    def test_consolidate_noops_with_nothing_new_to_consume(self, tmp_path, monkeypatch) -> None:
        """Nothing on disk, nothing pending → ``/consolidate`` noops.

        No content-bearing interim slot at N > 0: the fold's only content at
        this count.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.json() == {"status": "noop_no_interim_slots", "action": "full"}
        assert submitted == []

    def test_consolidate_drains_the_ring_in_manual_only_mode(self, tmp_path, monkeypatch) -> None:
        """Manual-only posture (``refresh_cadence: ""``, N > 0): ``/consolidate``
        with aged payload-bearing slots still dispatches and drains the ring.

        With no timer configured at all, a scheduled tick could never resolve
        this cycle — ``/consolidate`` is the only door that ever fires, and it
        does so on content alone.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=2, refresh_cadence="")
        for i in range(3):
            _make_interim_slot(
                state["config"].adapter_dir, f"2020010{i + 1}T0000", payload="weights"
            )

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.json() == {"status": "started_full", "action": "full"}
        assert _route_key_sources(submitted) == ["all_tiers"]

    def test_consolidate_noops_on_an_empty_ring_in_manual_only_mode(
        self, tmp_path, monkeypatch
    ) -> None:
        """Manual-only posture, empty ring → ``/consolidate`` noops."""
        state = _make_arbitrator_state(tmp_path, max_interim_count=2, refresh_cadence="")

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.json() == {"status": "noop_no_interim_slots", "action": "full"}
        assert submitted == []

    def test_consolidate_absorbs_a_leftover_interim_slot_at_count_zero(
        self, tmp_path, monkeypatch
    ) -> None:
        """A payload-bearing slot stranded by lowering ``max_interim_count`` to 0
        is still absorbed and reaped by ``/consolidate`` — no pending session
        needed.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=0)
        _make_interim_slot(state["config"].adapter_dir, "20260101T0000", payload="weights")

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.json() == {"status": "started_full", "action": "full"}
        assert _route_key_sources(submitted) == ["all_tiers"]

    def test_reconsolidate_reports_reconcile_and_narrows_the_key_source(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/reconsolidate`` is its own action, and the fold is told so.

        Same entry point and same status as ``/consolidate`` — the difference
        is the key source it binds, which is what leaves the interim slots
        alone.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "started_full", "action": "reconcile"}
        assert _route_key_sources(submitted) == ["main_tiers"]

    def test_reconsolidate_runs_with_nothing_new_to_consume(self, tmp_path, monkeypatch) -> None:
        """Nothing on disk, nothing pending → ``/reconsolidate`` still dispatches.

        Its input is the knowledge already stored; "nothing new" is not a reason
        to refuse it.  This is what it is for after a model/prompt/extraction
        change.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.json() == {"status": "started_full", "action": "reconcile"}
        assert _route_key_sources(submitted) == ["main_tiers"]

    def test_the_two_full_fold_doors_differ_only_in_the_key_source(
        self, tmp_path, monkeypatch
    ) -> None:
        """There is no manual flavour of a fold: same status, same entry point.

        The only thing that differs below the arbitrator is which keys the fold
        owns — and, following from that, whether the interim slots are reaped.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        for i in range(8):
            _make_interim_slot(
                state["config"].adapter_dir, f"202607{i + 1:02d}T0000", payload="weights"
            )

        client, submitted = _route_client(state, monkeypatch)
        collapse = client.post("/consolidate").json()
        rebuild = client.post("/reconsolidate").json()

        assert collapse["status"] == rebuild["status"] == "started_full"
        assert (collapse["action"], rebuild["action"]) == ("full", "reconcile")
        assert _route_key_sources(submitted) == ["all_tiers", "main_tiers"]

    def test_interim_route_absorbs_conversations_at_the_full_due_boundary(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/consolidate/interim`` is not re-decided when a full fold is due.

        Aged content-bearing slots past the full period → a scheduled tick
        would resolve FULL.  The operator asked for "absorb the recent
        conversations" directly, and gets exactly that.
        """
        import paramem.server.app as app_module

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, named_sessions=1, period_seconds=1
        )
        for i in range(8):
            _make_interim_slot(
                state["config"].adapter_dir, f"202001{i + 1:02d}T0000", payload="weights"
            )
        assert app_module._is_full_cycle_due(state["config"]) is True, (
            "fixture guard: a scheduled tick must resolve FULL here"
        )

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate/interim")

        assert resp.status_code == 200
        assert resp.json() == {"status": "started", "action": "interim"}
        assert submitted == [(app_module._extract_and_start_training, "started")]

    def test_interim_route_at_count_zero_reports_no_interim_tier(
        self, tmp_path, monkeypatch
    ) -> None:
        """At ``max_interim_count == 0`` there is no interim tier — the call is refused.

        A tier check, not a content check: the tier the request names does not
        exist, so there is no operation to run.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate/interim")

        assert resp.status_code == 200
        assert resp.json() == {"status": "noop_no_interim_tier", "action": "interim"}
        assert submitted == []

    def test_reconcile_is_the_only_door_the_content_gate_does_not_stop(
        self, tmp_path, monkeypatch
    ) -> None:
        """Nothing new to consume: every door noops except ``/reconsolidate``.

        ``/scheduled-tick`` resolves ``AUTO`` to INTERIM (no interim slots at
        all, so ``_is_full_cycle_due`` is False) and noops for lack of pending
        sessions; ``/consolidate`` requests ``FULL`` directly and noops for
        lack of a content-bearing interim slot — a DIFFERENT status, because
        it is a different action with a different input.
        ``/consolidate/interim`` requests ``INTERIM`` directly and noops the
        same way the tick did.  ``/reconsolidate`` is the one action exempt:
        its input (the main tiers' own stored keys) always exists.
        """
        import paramem.server.app as app_module

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        tick = client.post("/scheduled-tick").json()
        full = client.post("/consolidate").json()
        interim = client.post("/consolidate/interim").json()
        reconcile = client.post("/reconsolidate").json()

        assert tick == {"status": "noop_no_pending", "action": "interim"}
        assert full == {"status": "noop_no_interim_slots", "action": "full"}
        assert interim == {"status": "noop_no_pending", "action": "interim"}
        assert reconcile["status"] == "started_full"
        assert reconcile["action"] == "reconcile"
        assert len(submitted) == 1, "only /reconsolidate may have dispatched"
        fn, status = submitted[0]
        assert fn.func is app_module._run_full_consolidation_sync
        assert fn.args == ("main_tiers",)
        assert status == "started_full"

    def test_consolidate_route_ignores_a_stray_body(self, tmp_path, monkeypatch) -> None:
        """No route declares a body — a caller that posts one is not rejected for it.

        ``scripts/dev/probe_orphan_classification_live.py`` posts ``{}``; it must
        keep working.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        client, _submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate", json={})

        assert resp.status_code == 200
        assert resp.json()["status"] == "started_full"

    def test_reconsolidate_surfaces_a_deferral(self, tmp_path, monkeypatch) -> None:
        """A busy server defers: HTTP 200, ``deferred_*`` in ``status``, nothing submitted.

        The four doors report their outcome the same way — the status string,
        not the HTTP code, is where a consolidation outcome lives.
        """
        import time as _time

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["last_chat_monotonic"] = _time.monotonic() - 5  # debounce is 30 s

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "deferred_idle", "action": "reconcile"}
        assert submitted == []

    def test_scheduled_tick_stamps_the_cadence_but_consolidate_does_not(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/scheduled-tick`` advances the cadence stamp on dispatch; ``/consolidate`` never does.

        Same content-bearing interim slot on disk — ``/consolidate`` folds it
        directly (no deadline check), ``/scheduled-tick`` resolves ``AUTO``
        (here, to INTERIM, since no deadline is configured) and stamps because
        it IS the scheduled cycle.
        """
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, named_sessions=1, refresh_cadence="every 5h"
        )
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")
        state_dir = state["config"].paths.data / "state"
        seeded_stamp = time.time() - 6 * 3600  # outside the 5h window: due
        write_last_scheduled_run(state_dir, seeded_stamp)

        client, submitted = _route_client(state, monkeypatch)

        manual_resp = client.post("/consolidate")
        assert manual_resp.json()["status"] == "started_full"
        assert read_last_scheduled_run(state_dir) == seeded_stamp, (
            "a manual /consolidate dispatch must not move the cadence window"
        )

        tick_resp = client.post("/scheduled-tick")
        assert tick_resp.json()["status"] == "started"
        assert read_last_scheduled_run(state_dir) != seeded_stamp, (
            "the scheduled tick must advance the cadence stamp on dispatch"
        )
        assert len(submitted) == 2

    def test_consolidate_is_not_subject_to_the_catchup_not_due_gate(
        self, tmp_path, monkeypatch
    ) -> None:
        """A heartbeat wakeup not yet due blocks ``/scheduled-tick`` but not ``/consolidate``.

        The catch-up gate belongs to the systemd timer alone (``AUTO``); a
        directly requested ``FULL`` never consults the deadline machinery at
        all.
        """
        from paramem.server.schedule_state import write_last_scheduled_run

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="every 5h")
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")
        state_dir = state["config"].paths.data / "state"
        # Recent stamp -- inside the 5h window, so the tick is NOT yet due.
        write_last_scheduled_run(state_dir, time.time() - 60)

        client, submitted = _route_client(state, monkeypatch)

        tick_resp = client.post("/scheduled-tick")
        assert tick_resp.json()["status"] == "noop_not_due"

        manual_resp = client.post("/consolidate")
        assert manual_resp.json()["status"] == "started_full"
        assert len(submitted) == 1, "only the manual request may have dispatched"

    def test_only_scheduled_tick_ever_requests_auto(self, tmp_path, monkeypatch) -> None:
        """Structural pin: ``/scheduled-tick`` is the only door that ever passes
        ``AUTO`` to the arbitrator; the other three pass their own action, never
        ``AUTO``.

        Stands in for a runtime raise-on-mismatch guard: since ``AUTO`` is
        requested by exactly one caller, "action == AUTO with the wrong caller"
        is not a reachable runtime state to guard against — it is a property of
        which of the four routes was called, pinned here directly.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        actions_seen: list[ConsolidationAction] = []

        def _record_action(action):
            actions_seen.append(action)
            return "recorded", action

        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_dispatch_consolidation", _record_action)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/scheduled-tick")
        client.post("/consolidate")
        client.post("/consolidate/interim")
        client.post("/reconsolidate")

        assert actions_seen == [
            ConsolidationAction.AUTO,
            ConsolidationAction.FULL,
            ConsolidationAction.INTERIM,
            ConsolidationAction.RECONCILE,
        ]
        assert actions_seen.count(ConsolidationAction.AUTO) == 1, (
            "AUTO must be requested by exactly one door: /scheduled-tick"
        )


# ---------------------------------------------------------------------------
# TestFullConsolidationFoldEntry
# ---------------------------------------------------------------------------


class TestFullConsolidationFoldEntry:
    """_run_full_consolidation_sync drives the fold entry with its venue, key source
    and fold inputs."""

    def _run_sync(self, state: dict, monkeypatch, keys_from: str = "all_tiers") -> None:
        """Run _run_full_consolidation_sync with an inlined BackgroundTrainer."""
        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)
        mock_bt = MagicMock()
        mock_bt.abort_requested = False
        # submit() calls the closure synchronously so state can be inspected after.
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync(keys_from)

    def test_fold_called_with_venue_key_source_and_fold_inputs_only(
        self, monkeypatch, tmp_path
    ) -> None:
        """The fold receives the configured venue, its key source and its collaborators.

        The kwarg set is pinned by exact equality — the mode, the key source,
        the config-derived ``consume_pending`` decision and the two
        collaborators the fold runs on.
        """
        state = _make_dispatch_state(consolidation_mode="train", tmp_path=tmp_path)

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        loop = state["consolidation_loop"]
        loop.consolidate.assert_called_once()
        args, kwargs = loop.consolidate.call_args
        assert args == (), "the fold entry is keyword-only"
        assert set(kwargs) == {"mode", "keys_from", "trainer", "router", "consume_pending"}, (
            f"unexpected fold-entry kwargs: {sorted(kwargs)}"
        )
        assert kwargs["mode"] == "train"
        assert kwargs["keys_from"] == "all_tiers"
        assert kwargs["consume_pending"] is False, (
            "max_interim_count=7 → the fold must not consume pending sessions"
        )

    def test_main_tiers_fold_forwards_its_key_source(self, monkeypatch, tmp_path) -> None:
        """A reconcile reaches the fold as ``keys_from="main_tiers"``."""
        state = _make_dispatch_state(consolidation_mode="train", tmp_path=tmp_path)

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch, keys_from="main_tiers")

        _, kwargs = state["consolidation_loop"].consolidate.call_args
        assert kwargs["keys_from"] == "main_tiers"

    def test_main_tiers_fold_never_consumes_pending_sessions(self, monkeypatch, tmp_path) -> None:
        """At ``max_interim_count == 0`` the absorbing fold consumes pending sessions.

        A main-tiers fold must not, whatever the count says: it leaves the
        pending conversations pending, so it must not run the extraction
        pre-stage either.
        """
        import paramem.server.app as app_module

        extract_calls: list[object] = []

        def _record_extract(loop, *, lock_held):
            extract_calls.append(loop)
            raise AssertionError("the extraction pre-stage must not run for a main-tiers fold")

        absorbing = _make_dispatch_state(
            consolidation_mode="train", max_interim_count=0, tmp_path=tmp_path
        )
        reconciling = _make_dispatch_state(
            consolidation_mode="train", max_interim_count=0, tmp_path=tmp_path
        )

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
            patch.object(app_module, "_extract_pending_sessions", _record_extract),
        ):
            self._run_sync(reconciling, monkeypatch, keys_from="main_tiers")

        _, kwargs = reconciling["consolidation_loop"].consolidate.call_args
        assert kwargs["consume_pending"] is False
        assert extract_calls == []

        # Same config, absorbing fold: consume_pending is back on, so the
        # False above is the key source's doing and not the config's.
        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
            patch.object(
                app_module,
                "_extract_pending_sessions",
                MagicMock(
                    return_value=app_module._PendingExtraction(
                        episodic_rels=[],
                        procedural_rels=[],
                        session_ids=[],
                        failed_session_ids=set(),
                        speaker_ids=[],
                        evicted_voice=False,
                        aborted=None,
                    )
                ),
            ),
        ):
            self._run_sync(absorbing, monkeypatch, keys_from="all_tiers")

        _, kwargs = absorbing["consolidation_loop"].consolidate.call_args
        assert kwargs["consume_pending"] is True

    def test_simulate_mode_uses_the_same_entry(self, monkeypatch, tmp_path) -> None:
        """Simulate mode routes through the identical call — only ``mode`` differs."""
        state = _make_dispatch_state(consolidation_mode="simulate", tmp_path=tmp_path)

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        loop = state["consolidation_loop"]
        loop.consolidate.assert_called_once()
        _, kwargs = loop.consolidate.call_args
        assert kwargs["mode"] == "simulate"
        assert kwargs["consume_pending"] is False

    def test_empty_tiers_rebuilt_is_a_noop_terminal(self, monkeypatch, tmp_path) -> None:
        """tiers_rebuilt == [] ends the cycle as a noop for every caller.

        The flag that used to exempt the on-demand fold from this guard is gone:
        an empty rebuild is a noop no matter who dispatched it, and the
        ``consolidating`` flag is cleared on the way out.
        """
        state = _make_dispatch_state(tmp_path=tmp_path)
        state["consolidating"] = True  # set by the dispatcher before submit

        with (
            patch("paramem.server.consolidation._save_key_metadata") as mock_save_meta,
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        # The noop terminal returns before the key-metadata persist.
        mock_save_meta.assert_not_called()
        assert state["consolidating"] is False, (
            "_state['consolidating'] must be cleared after the fold completes"
        )
