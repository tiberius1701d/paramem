"""Tests for the server-side consolidation dispatch infrastructure.

All tests are CPU-only, no model load required.  The consolidation-loop and
BackgroundTrainer are mocked so the implementation-level dispatch paths can be
verified in isolation.

Coverage:
- ``_consolidation_dispatch_guards`` shared guard helper
- ``_dispatch_consolidation`` — the arbitrator: action resolution, the single
  content gate, the executor submission ritual, and the concurrency guard
- ``_run_full_consolidation_sync`` noop terminal: an empty ``tiers_rebuilt``
  ends the cycle as a noop, and the sessions consumed by the pre-stage are
  still retired so they cannot accumulate unboundedly.

The fold itself carries no flags: the arbitrator decides whether there is
anything to consolidate, and ``loop.consolidate(mode=...)`` then does what it
is told.
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
        "recall_per_tier": {},
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
    """
    from paramem.server.session_buffer import SessionBuffer

    cfg = MagicMock()
    cfg.consolidation.mode = consolidation_mode
    cfg.consolidation.max_interim_count = max_interim_count
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.consolidation.training_idle_debounce_s = 30
    cfg.consolidation.orphan_retirement_seconds = None
    cfg.consolidation.retain_sessions = False
    # Manual-only period → _full_consolidation_overdue_key returns None (no
    # incident I/O in the dispatch path).
    cfg.consolidation.consolidation_period_seconds = None
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


def _dispatch(state, action, *, apply_schedule_gate=True, monkeypatch=None):
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
        status, resolved = app_module._dispatch_consolidation(
            action, apply_schedule_gate=apply_schedule_gate
        )
    return status, resolved, spy, due_calls


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

        At max_interim_count==0 the interim path never runs — the full path's
        content gate is the ONLY place session triage happens.  Before this
        step, nothing retired UNIDENTIFIABLE/expired-HOLDABLE sessions at N=0;
        they accumulated in the buffer forever.  This is the regression test
        for that leak: it must fail if a future change moves the triage behind
        the ``force`` check or reorders the FULL disjuncts to skip it when no
        interim slots exist.
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
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert spy.submitted == [app_module._run_full_consolidation_sync]
        assert state["consolidating"] is True

    def test_full_due_with_content_bearing_interims_dispatches(self, tmp_path, monkeypatch) -> None:
        """N>0, full cycle due, content-bearing interim slots → dispatches."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=2)
        adapter_dir = state["config"].adapter_dir
        # n=3, N=2 → (n-1) % N == 0 → count-primary due.
        for i in range(3):
            _make_interim_slot(adapter_dir, f"2026070{i + 1}T0000", payload="weights")

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert spy.call_count == 1
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
            status, resolved = app_module._dispatch_consolidation(
                ConsolidationAction.AUTO, apply_schedule_gate=True
            )

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
            status, _resolved = app_module._dispatch_consolidation(
                ConsolidationAction.AUTO, apply_schedule_gate=True
            )

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
            status, _resolved = app_module._dispatch_consolidation(
                ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert status == "noop_no_pending"
        assert spy.call_count == 0

    def test_forced_full_runs_with_no_interims_and_no_pending(self, tmp_path, monkeypatch) -> None:
        """An explicitly requested FULL runs on an empty store — that is what forcing means.

        Its input is the existing adapter weights, which are content by
        definition.  The schedule gate is never consulted.
        """
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state,
            ConsolidationAction.FULL,
            apply_schedule_gate=False,
            monkeypatch=monkeypatch,
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert spy.submitted == [app_module._run_full_consolidation_sync]
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit FULL"

    def test_fold_entry_takes_mode_and_fold_inputs_only(self, tmp_path) -> None:
        """The arbitrator's bypass decision stays in the arbitrator.

        The fold entry's parameter set is pinned by exact set equality, so any
        caller-intent parameter leaking down from the dispatch layer fails here.
        """
        import inspect

        from paramem.training.consolidation import ConsolidationLoop

        params = inspect.signature(ConsolidationLoop.consolidate).parameters
        assert set(params) == {
            "self",
            "mode",
            "consume_pending",
            "trainer",
            "router",
            "recall_sanity_threshold",
        }, f"unexpected fold-entry parameters: {sorted(params)}"

    def test_interim_with_no_pending_is_not_bypassable(self, tmp_path, monkeypatch) -> None:
        """An explicitly requested INTERIM with zero pending sessions noops.

        Force does not apply to INTERIM: its only input is pending sessions, and
        minting an empty interim slot burns a ring slot and a training run for
        nothing.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, resolved, spy, due_calls = _dispatch(
            state,
            ConsolidationAction.INTERIM,
            apply_schedule_gate=False,
            monkeypatch=monkeypatch,
        )

        assert status == "noop_no_pending"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.call_count == 0
        assert due_calls == [], "_is_full_cycle_due must not be consulted for an explicit INTERIM"

    def test_interim_with_only_unattributable_sessions_returns_noop_no_named(
        self, tmp_path, monkeypatch
    ) -> None:
        """Pending but unattributable → noop_no_named, and they are retired."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, anon_sessions=2)
        status, _resolved, spy, _ = _dispatch(
            state,
            ConsolidationAction.INTERIM,
            apply_schedule_gate=False,
            monkeypatch=monkeypatch,
        )

        assert status == "noop_no_named"
        assert spy.call_count == 0
        assert state["session_buffer"].pending_facts() == [], (
            "UNIDENTIFIABLE sessions must be retired, not left to accumulate"
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
            apply_schedule_gate=False,
            monkeypatch=monkeypatch,
        )

        assert status == "started"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._extract_and_start_training]

    def test_interim_at_count_zero_is_refused(self, tmp_path, monkeypatch) -> None:
        """An explicit INTERIM at max_interim_count==0 → noop_no_interim_tier.

        There is no interim tier at N=0; the request is meaningless.  NAMED
        sessions are pending here, so only the tier check can stop it.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        status, _resolved, spy, _ = _dispatch(
            state,
            ConsolidationAction.INTERIM,
            apply_schedule_gate=False,
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
        input (three content-bearing interim slots) must produce the same
        outcome in simulate and in train.
        """
        from paramem.server.app import ConsolidationAction

        payload = "graph" if consolidation_mode == "simulate" else "weights"
        state = _make_arbitrator_state(
            tmp_path, consolidation_mode=consolidation_mode, max_interim_count=2
        )
        for i in range(3):
            _make_interim_slot(state["config"].adapter_dir, f"2026070{i + 1}T0000", payload=payload)

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
            apply_schedule_gate=False,
            monkeypatch=monkeypatch,
        )

        assert status == "deferred_idle"
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# TestStampPredicate — a dispatch advances the schedule stamp iff it consumes
# something.  A forced FULL that passed the content gate only because it was
# forced (nothing on disk, nothing pending) is a pure idempotent re-groom and
# must not perturb the cadence.
# ---------------------------------------------------------------------------


class TestStampPredicate:
    """``_stamp_scheduled_run`` fires iff the dispatch has real content to consume."""

    def _dispatch_and_track_stamp(
        self, state, action, *, apply_schedule_gate, monkeypatch
    ) -> "tuple[str, object, int]":
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
            status, resolved = app_module._dispatch_consolidation(
                action, apply_schedule_gate=apply_schedule_gate
            )
        return status, resolved, len(stamp_calls)

    def test_forced_full_with_content_bearing_interims_stamps(self, tmp_path, monkeypatch) -> None:
        """Forced FULL, content-bearing interim slots present → dispatches AND stamps.

        The fold absorbs and purges every interim slot on the success path
        (``unload_interim_adapters`` in ``paramem/training/consolidation.py``),
        so this fold does exactly what the scheduled cycle would have done.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=2)
        for i in range(2):
            _make_interim_slot(
                state["config"].adapter_dir, f"2026070{i + 1}T0000", payload="weights"
            )

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.FULL, apply_schedule_gate=False, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert stamp_calls == 1

    def test_forced_full_with_zero_interims_and_zero_pending_does_not_stamp(
        self, tmp_path, monkeypatch
    ) -> None:
        """Forced FULL, nothing on disk and nothing pending → dispatches but does NOT stamp.

        This is ``/reconsolidate``'s "rebuild from stored knowledge" case: it
        consumes nothing, so it must not perturb the cadence.  Checked two
        ways — the wrapped ``_stamp_scheduled_run`` is never called, AND
        (using a non-calendar-exact cadence, where the stamp is a real
        on-disk write) the persisted ``last_scheduled_run`` is byte-identical
        before and after.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="every 5h")
        state_dir = state["config"].paths.data / "state"
        seeded_stamp = time.time() - 6 * 3600
        write_last_scheduled_run(state_dir, seeded_stamp)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.FULL, apply_schedule_gate=False, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert stamp_calls == 0, "_stamp_scheduled_run must not be called for a forced no-op FULL"
        assert read_last_scheduled_run(state_dir) == seeded_stamp, (
            "the persisted last_scheduled_run must be unchanged by a forced FULL "
            "that consumed nothing"
        )

    def test_forced_full_with_zero_interims_but_named_pending_stamps(
        self, tmp_path, monkeypatch
    ) -> None:
        """Forced FULL, no interims but NAMED pending sessions → dispatches AND stamps.

        The fold will consume them (directly, at ``max_interim_count == 0``,
        or on the next interim tick otherwise) — either way this dispatch had
        real input.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=0, named_sessions=1, refresh_cadence="every 5h"
        )
        state_dir = state["config"].paths.data / "state"

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.FULL, apply_schedule_gate=False, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert stamp_calls == 1

        from paramem.server.schedule_state import read_last_scheduled_run

        assert read_last_scheduled_run(state_dir) is not None

    def test_auto_resolved_full_still_stamps(self, tmp_path, monkeypatch) -> None:
        """AUTO→FULL is never forced; when it dispatches it always has real
        content (the gate requires it), so it always stamps — unchanged.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=2, refresh_cadence="every 5h")
        for i in range(3):
            _make_interim_slot(
                state["config"].adapter_dir, f"2026070{i + 1}T0000", payload="weights"
            )

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.AUTO, apply_schedule_gate=False, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert stamp_calls == 1

    def test_interim_still_stamps(self, tmp_path, monkeypatch) -> None:
        """INTERIM is unchanged: it always stamps when it dispatches — it
        cannot reach dispatch without NAMED pending sessions (the content
        gate refuses an empty interim tick even when explicitly requested).
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, named_sessions=1, refresh_cadence="every 5h"
        )

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.INTERIM, apply_schedule_gate=False, monkeypatch=monkeypatch
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


class TestConsolidationRoutes:
    """The four consolidation routes: intent → arbitrator call → status/action."""

    def test_reconsolidate_rebuilds_main_memory_when_the_schedule_would_not(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/reconsolidate`` runs the full fold where the schedule says "interim".

        One content-bearing interim slot at N=7: ``_is_full_cycle_due`` is False,
        so an AUTO dispatch would resolve INTERIM.  The intent "rebuild main
        memory from stored knowledge" must not be re-decided by the schedule.
        """
        import paramem.server.app as app_module

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")
        assert app_module._is_full_cycle_due(state["config"]) is False, (
            "fixture guard: AUTO must resolve INTERIM here for this test to mean anything"
        )

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "started_full", "action": "full"}
        assert submitted == [(app_module._run_full_consolidation_sync, "started_full")]

    def test_reconsolidate_runs_with_nothing_new_to_consume(self, tmp_path, monkeypatch) -> None:
        """Nothing on disk, nothing pending → ``/reconsolidate`` still dispatches.

        Its input is the knowledge already stored; "nothing new" is not a reason
        to refuse it.  This is what it is for after a model/prompt/extraction
        change.
        """
        import paramem.server.app as app_module

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.json() == {"status": "started_full", "action": "full"}
        assert submitted == [(app_module._run_full_consolidation_sync, "started_full")]

    def test_reconsolidate_status_is_indistinguishable_from_a_scheduled_full(
        self, tmp_path, monkeypatch
    ) -> None:
        """A forced full fold reports ``started_full`` — the same family as a scheduled one.

        There is no manual/forced flavour of a fold: the telemetry, the outputs
        and the status string are the scheduled ones.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        for i in range(8):
            _make_interim_slot(
                state["config"].adapter_dir, f"202607{i + 1:02d}T0000", payload="weights"
            )

        client, _submitted = _route_client(state, monkeypatch)
        scheduled = client.post("/consolidate").json()
        forced = client.post("/reconsolidate").json()

        assert scheduled == forced == {"status": "started_full", "action": "full"}

    def test_interim_route_absorbs_conversations_at_the_full_due_boundary(
        self, tmp_path, monkeypatch
    ) -> None:
        """``/consolidate/interim`` bypasses the schedule gate when a full fold is due.

        Eight content-bearing slots at N=7 → ``_is_full_cycle_due`` is True, so
        an AUTO dispatch would resolve FULL.  The operator asked for "absorb the
        recent conversations", and gets exactly that.
        """
        import paramem.server.app as app_module

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        for i in range(8):
            _make_interim_slot(
                state["config"].adapter_dir, f"202607{i + 1:02d}T0000", payload="weights"
            )
        assert app_module._is_full_cycle_due(state["config"]) is True, (
            "fixture guard: AUTO must resolve FULL here for this test to mean anything"
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

        NAMED sessions are pending, so only the missing tier can stop it.  The
        operator is told which, and nothing is submitted.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate/interim")

        assert resp.status_code == 200
        assert resp.json() == {"status": "noop_no_interim_tier", "action": "interim"}
        assert submitted == []

    def test_interim_route_with_nothing_pending_is_a_noop(self, tmp_path, monkeypatch) -> None:
        """No pending sessions → ``noop_no_pending``; the content gate is not bypassable."""
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate/interim")

        assert resp.json() == {"status": "noop_no_pending", "action": "interim"}
        assert submitted == []

    @pytest.mark.parametrize(
        ("interims", "expected_status", "expected_action"),
        [(1, "started", "interim"), (8, "started_full", "full")],
    )
    def test_consolidate_route_still_resolves_via_the_schedule(
        self, tmp_path, monkeypatch, interims, expected_status, expected_action
    ) -> None:
        """``/consolidate`` (no body) is unchanged: ``_is_full_cycle_due`` decides.

        Every existing caller posts no body and keeps working — the schedule
        picks the action and ``action`` in the response reports which one ran.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        for i in range(interims):
            _make_interim_slot(
                state["config"].adapter_dir, f"2026070{i + 1:02d}T0000", payload="weights"
            )

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": expected_status, "action": expected_action}
        assert len(submitted) == 1

    def test_consolidate_route_ignores_a_stray_body(self, tmp_path, monkeypatch) -> None:
        """No route declares a body — a caller that posts one is not rejected for it.

        ``scripts/dev/probe_orphan_classification_live.py`` posts ``{}``; it must
        keep working.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)

        client, _submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate", json={})

        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

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
        assert resp.json() == {"status": "deferred_idle", "action": "full"}
        assert submitted == []


# ---------------------------------------------------------------------------
# TestFullConsolidationFoldEntry
# ---------------------------------------------------------------------------


class TestFullConsolidationFoldEntry:
    """_run_full_consolidation_sync drives the fold entry with its mode and fold inputs."""

    def _run_sync(self, state: dict, monkeypatch) -> None:
        """Run _run_full_consolidation_sync with an inlined BackgroundTrainer."""
        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)
        mock_bt = MagicMock()
        mock_bt.abort_requested = False
        # submit() calls the closure synchronously so state can be inspected after.
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync()

    def test_fold_called_with_mode_and_fold_inputs_only(self, monkeypatch, tmp_path) -> None:
        """The fold receives the configured mode and the collaborators it trains with.

        The kwarg set is pinned by exact equality — the mode, the config-derived
        ``consume_pending`` decision and the two collaborators the fold runs on.
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
        assert set(kwargs) == {"mode", "trainer", "router", "consume_pending"}, (
            f"unexpected fold-entry kwargs: {sorted(kwargs)}"
        )
        assert kwargs["mode"] == "train"
        assert kwargs["consume_pending"] is False, (
            "max_interim_count=7 → the fold must not consume pending sessions"
        )

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
