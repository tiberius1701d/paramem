"""Tests for the server-side consolidation dispatch infrastructure.

All tests are CPU-only, no model load required.  The consolidation-loop and
BackgroundTrainer are mocked so the implementation-level dispatch paths can be
verified in isolation.

Coverage:
- ``_consolidation_dispatch_guards`` shared guard helper
- ``_dispatch_consolidation`` — the arbitrator: one action per door, the
  schedule's three decisions (catch-up gate, AUTO resolution, content gate)
  applied to ``AUTO`` alone, the executor submission ritual, and the
  concurrency guard
- ``_run_full_consolidation_sync`` noop terminal: an empty ``tiers_rebuilt``
  ends the cycle as a noop, and the sessions consumed by the pre-stage are
  still retired so they cannot accumulate unboundedly.

The fold itself carries no caller intent: the arbitrator decides whether a
scheduled tick has anything to consolidate, and
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
        refresh_cadence: ``config.consolidation.refresh_cadence``.  The
            default ("12h") is calendar-exact — ``heartbeat_seconds()`` is
            ``None``, so ``_stamp_scheduled_run`` no-ops and these tests
            exercise the arbitrator, not the scheduler.  Tests that need to
            observe the PERSISTED stamp (rather than just whether
            ``_stamp_scheduled_run`` was called) must pass a non-calendar-exact
            value (e.g. ``"every 5h"``).
        period_seconds: ``config.consolidation.consolidation_period_seconds`` —
            the full-fold period the deadline is measured against.  ``None``
            (the default) is a manual-only cadence: no deadline, so
            ``_is_full_cycle_due`` is False for any interim ring and
            ``_full_consolidation_overdue_key`` returns ``None`` (no incident
            I/O in the dispatch path).  A test that needs AUTO to resolve FULL
            passes a small value together with an aged interim slot.
    """
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
    counts the ``_is_full_cycle_due`` invocations (the gate must be consulted
    exactly once, and only on AUTO).
    """
    import paramem.server.app as app_module

    spy = _ExecutorSpy()
    due_calls: list[bool] = []
    _real_due = app_module._is_full_cycle_due

    def _counting_due(config):
        result = _real_due(config)
        due_calls.append(result)
        return result

    monkeypatch.setattr(app_module, "_state", state)
    monkeypatch.setattr(app_module, "_is_full_cycle_due", _counting_due)
    monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
    with patch("asyncio.get_running_loop", return_value=spy.loop):
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

    def test_payload_less_interim_dirs_do_not_satisfy_the_content_gate(
        self, tmp_path, monkeypatch
    ) -> None:
        """Payload-less interim DIRECTORIES are not content.

        A slot whose payload write never landed holds nothing to fold.  Forcing
        the full path (via a stubbed schedule gate) with only such shells on
        disk and no pending sessions must noop, not dispatch.
        """
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=2)
        adapter_dir = state["config"].adapter_dir
        for i in range(3):
            _make_interim_slot(adapter_dir, f"2026070{i + 1}T0000", payload=None)

        spy = _ExecutorSpy()
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        # The schedule gate is stubbed True so the ONLY thing standing between
        # the tick and a full GPU retrain is the content gate.
        monkeypatch.setattr(app_module, "_is_full_cycle_due", lambda config: True)
        with patch("asyncio.get_running_loop", return_value=spy.loop):
            status, resolved = app_module._dispatch_consolidation(ConsolidationAction.AUTO)

        assert status == "noop_no_pending"
        assert resolved is ConsolidationAction.FULL
        assert spy.call_count == 0

    def test_payload_bearing_interim_dir_does_satisfy_the_content_gate(
        self, tmp_path, monkeypatch
    ) -> None:
        """The same slots WITH the venue payload do satisfy the gate."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=2)
        adapter_dir = state["config"].adapter_dir
        for i in range(3):
            _make_interim_slot(adapter_dir, f"2026070{i + 1}T0000", payload="weights")

        spy = _ExecutorSpy()
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        monkeypatch.setattr(app_module, "_is_full_cycle_due", lambda config: True)
        with patch("asyncio.get_running_loop", return_value=spy.loop):
            status, _resolved = app_module._dispatch_consolidation(ConsolidationAction.AUTO)

        assert status == "started_full"
        assert spy.call_count == 1

    def test_wrong_venue_payload_does_not_satisfy_the_gate(self, tmp_path, monkeypatch) -> None:
        """A train-venue payload is not content in simulate mode."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, consolidation_mode="simulate", max_interim_count=2)
        adapter_dir = state["config"].adapter_dir
        for i in range(3):
            _make_interim_slot(adapter_dir, f"2026070{i + 1}T0000", payload="weights")

        spy = _ExecutorSpy()
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        monkeypatch.setattr(app_module, "_is_full_cycle_due", lambda config: True)
        with patch("asyncio.get_running_loop", return_value=spy.loop):
            status, _resolved = app_module._dispatch_consolidation(ConsolidationAction.AUTO)

        assert status == "noop_no_pending"
        assert spy.call_count == 0

    def test_full_runs_with_no_interims_and_no_pending(self, tmp_path, monkeypatch) -> None:
        """An explicitly requested FULL runs on an empty store.

        Its input is the existing adapter weights, which are content by
        definition.  Neither the schedule gate nor the content gate is
        consulted — an operator door is a deliberate request.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state,
            ConsolidationAction.FULL,
            monkeypatch=monkeypatch,
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_key_sources(spy) == ["all_tiers"]
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit FULL"

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
            state,
            ConsolidationAction.RECONCILE,
            monkeypatch=monkeypatch,
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE
        assert _submitted_full_fold_key_sources(spy) == ["main_tiers"]
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit RECONCILE"

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
            "recall_sanity_threshold",
        }, f"unexpected fold-entry parameters: {sorted(params)}"

    def test_interim_with_no_pending_still_dispatches(self, tmp_path, monkeypatch) -> None:
        """An explicitly requested INTERIM with zero pending sessions runs.

        The content gate belongs to the schedule; a door the operator opened is
        not refused for "nothing is waiting".  It is not a burnt ring slot
        either: with an empty batch the extraction phase finds no relations and
        returns before any slot is minted (``_extract_and_start_training``).
        """
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state,
            ConsolidationAction.INTERIM,
            monkeypatch=monkeypatch,
        )

        assert status == "started"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._extract_and_start_training]
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit INTERIM"

    def test_unattributable_sessions_are_retired_on_every_door(self, tmp_path, monkeypatch) -> None:
        """Orphan retirement is a pre-stage, not part of the content gate.

        Every door bypasses the content gate, so if triage lived inside it, an
        operator-only deployment would never retire an unattributable session
        again.  Pinned on each door in turn.
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
            state,
            ConsolidationAction.INTERIM,
            monkeypatch=monkeypatch,
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
            state,
            ConsolidationAction.INTERIM,
            monkeypatch=monkeypatch,
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

    def test_idle_debounce_applies_to_an_explicit_full(self, tmp_path, monkeypatch) -> None:
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
            state,
            ConsolidationAction.FULL,
            monkeypatch=monkeypatch,
        )

        assert status == "deferred_idle"
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# TestStampPredicate — the schedule stamp belongs to the schedule.  A manual
# run does not move the cadence window; the next scheduled tick still has its
# content gate and noops on its own if the manual run consumed everything.
# ---------------------------------------------------------------------------


class TestStampPredicate:
    """``_stamp_scheduled_run`` fires iff the dispatch came from the schedule."""

    def _dispatch_and_track_stamp(self, state, action, *, monkeypatch) -> "tuple[str, object, int]":
        """Run the arbitrator, counting real (unmocked) ``_stamp_scheduled_run`` calls.

        The real function still runs (so the persisted stamp can be checked
        separately) — this only counts invocations.
        """
        import paramem.server.app as app_module

        spy = _ExecutorSpy()
        stamp_calls: list[object] = []
        _real_stamp = app_module._stamp_scheduled_run

        def _counting_stamp(config):
            stamp_calls.append(config)
            return _real_stamp(config)

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_stamp_scheduled_run", _counting_stamp)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        with patch("asyncio.get_running_loop", return_value=spy.loop):
            status, resolved = app_module._dispatch_consolidation(action)
        return status, resolved, len(stamp_calls)

    @pytest.mark.parametrize("action_name", ["FULL", "INTERIM", "RECONCILE"])
    def test_no_manual_door_moves_the_cadence_window(
        self, tmp_path, monkeypatch, action_name
    ) -> None:
        """Every operator door dispatches WITHOUT stamping.

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
# TestConsolidationRoutes — the operator surface: four intent-named, body-less
# doors onto the one arbitrator.  Each door names an intent; none of them
# exposes an internal knob (no mode, no force, no request body at all).
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

    def test_consolidate_collapses_the_interims_whatever_the_schedule_says(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/consolidate`` runs the full fold where the schedule says "interim".

        One content-bearing interim slot at N=7 with no deadline:
        ``_is_full_cycle_due`` is False, so a scheduled tick would resolve
        INTERIM.  The operator asked to collapse the interims now, and gets
        exactly that — this is the defect the door fixes: with
        ``refresh_cadence: ""`` an AUTO-dispatching ``/consolidate`` could never
        reach a full cycle at all.
        """
        import paramem.server.app as app_module

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")
        assert app_module._is_full_cycle_due(state["config"]) is False, (
            "fixture guard: a scheduled tick must resolve INTERIM here"
        )

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "started_full", "action": "full"}
        assert _route_key_sources(submitted) == ["all_tiers"]

    def test_consolidate_runs_with_nothing_new_to_consume(self, tmp_path, monkeypatch) -> None:
        """Nothing on disk, nothing pending → ``/consolidate`` still dispatches.

        The content gate belongs to the schedule; a deliberate request is not
        refused for "nothing new".
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

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

        Aged content-bearing slots past the full period → ``_is_full_cycle_due``
        is True, so a scheduled tick would resolve FULL.  The operator asked for
        "absorb the recent conversations", and gets exactly that.
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

    def test_scheduled_tick_is_the_only_door_the_content_gate_stops(
        self, tmp_path, monkeypatch
    ) -> None:
        """Nothing new to consume: the tick noops, the operator doors run."""
        import paramem.server.app as app_module

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        tick = client.post("/scheduled-tick").json()

        assert tick == {"status": "noop_no_pending", "action": "interim"}
        assert submitted == []

        assert client.post("/consolidate").json()["status"] == "started_full"
        state["consolidating"] = False
        assert client.post("/consolidate/interim").json()["status"] == "started"
        assert submitted[0][0].func is app_module._run_full_consolidation_sync
        assert submitted[1][0] is app_module._extract_and_start_training

    def test_consolidate_route_ignores_a_stray_body(self, tmp_path, monkeypatch) -> None:
        """No route declares a body — a caller that posts one is not rejected for it.

        ``scripts/dev/probe_orphan_classification_live.py`` posts ``{}``; it must
        keep working.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)

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
