"""Tests for the server-side consolidation dispatch infrastructure.

All tests are CPU-only, no model load required.  The consolidation-loop and
BackgroundTrainer are mocked so the implementation-level dispatch paths can be
verified in isolation.

Coverage:
- ``_consolidation_dispatch_guards`` shared guard helper
- ``_dispatch_consolidation`` — the arbitrator: ``AUTO`` is requested ONLY by
  ``/scheduled-tick`` (deadline resolution via ``_is_full_cycle_due``, the
  catch-up gate, and the cadence stamp are its business alone).  ``FULL``,
  ``INTERIM``, and ``RECONCILE`` are each requestable directly
  (``/consolidate``, ``/consolidate/interim``, ``/reconsolidate``) as well as
  ``FULL``/``INTERIM`` via ``AUTO``'s resolution, and the content gate applies
  to every one of them — a manual door drops only the TIME condition, never
  the CONTENT condition.  ``RECONCILE`` is a full consolidation whose input
  excludes pending sessions; its content is any active key already held by
  any tier, main or interim, so it is turned away only by an empty store,
  never by the absence of new interim/pending material.  Also covers the
  executor submission ritual and the concurrency guard.
- ``_run_full_consolidation_sync`` noop terminal: an empty ``tiers_rebuilt``
  ends the cycle as a noop, and the sessions consumed by the pre-stage are
  still retired so they cannot accumulate unboundedly.

The fold itself carries no caller intent: the arbitrator decides whether a
dispatch has anything to consolidate, and
``loop.consolidate(mode=..., event=...)`` then does what it is told with the
venue and door name it was handed — a full fold and a reconcile run the
identical fold topology.

The adapter_manifest_status-driven ``deferred_tier_unverified`` gate is
covered in ``tests/server/test_startup_validator.py``, not here.
``_finish_resumed_event``'s ``run_build_and_publish``-summary threading has
no dedicated suite currently. Every pending-record test in this file builds
its ``StageLedger`` through the shared
``tests.server._state_builders._write_pending_ledger`` fixture, never
inline.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from tests.server._state_builders import _write_pending_ledger

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
    store=None,
    consolidate_return: "dict | None" = None,
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
        store: Override for ``loop.store`` — a real ``MemoryStore`` (for a
            test that needs real entry content) or a purpose-built
            ``MagicMock`` (e.g. a spy-able ``.swap()``).  Defaults to a bare
            unconfigured ``MagicMock``.
        consolidate_return: Override for ``loop.consolidate.return_value``.
            Defaults to a successful noop-ish result with ``tiers_rebuilt=[]``.
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
    if store is not None:
        mock_loop.store = store
    # Default fold return: successful noop-ish result with tiers_rebuilt=[].
    mock_loop.consolidate.return_value = (
        consolidate_return
        if consolidate_return is not None
        else {
            "tiers_rebuilt": [],
            "completed": False,
            "aborted": False,
        }
    )

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
# TestConsolidationLoopStoreOverride — the store-independent-resume seam:
# get_or_create_consolidation_loop(state, store=...) and
# _run_stage_b_cycle(..., store=...) thread an optional store override
# through to loop construction, reachable in production ONLY from the
# pending-event resume (paramem/server/app.py::_run_pending_event_resume).
# Every other caller passes no override and is provably unaffected.
# ---------------------------------------------------------------------------


class TestConsolidationLoopStoreOverride:
    """``get_or_create_consolidation_loop`` (paramem.server.consolidation) —
    the ONE get-or-create the whole tree shares, taking ``state`` rather
    than a bare ``config``."""

    def test_store_override_used_only_on_a_fresh_construction(self, monkeypatch) -> None:
        """No cached loop -> a fresh construction reads the override, not
        ``_state["memory_store"]``."""
        import paramem.server.app as app_module

        sentinel_store = object()
        default_store = object()
        seen: list = []

        def _fake_create(model, tokenizer, config, store, **kwargs):
            seen.append(store)
            return MagicMock()

        state = {
            "consolidation_loop": None,
            "config": MagicMock(),
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "memory_store": default_store,
        }
        monkeypatch.setattr(app_module, "_state", state)
        import paramem.server.consolidation as consolidation_module

        monkeypatch.setattr(consolidation_module, "create_consolidation_loop", _fake_create)

        loop = app_module.get_or_create_consolidation_loop(state, store=sentinel_store)

        assert seen == [sentinel_store]
        assert state["consolidation_loop"] is loop

    def test_no_override_falls_through_to_state_memory_store(self, monkeypatch) -> None:
        """Every ordinary caller (no ``store=`` kwarg) sees unchanged
        behaviour — ``_state["memory_store"]`` is what a fresh loop is
        built against."""
        import paramem.server.app as app_module

        default_store = object()
        seen: list = []

        def _fake_create(model, tokenizer, config, store, **kwargs):
            seen.append(store)
            return MagicMock()

        state = {
            "consolidation_loop": None,
            "config": MagicMock(),
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "memory_store": default_store,
        }
        monkeypatch.setattr(app_module, "_state", state)
        import paramem.server.consolidation as consolidation_module

        monkeypatch.setattr(consolidation_module, "create_consolidation_loop", _fake_create)

        app_module.get_or_create_consolidation_loop(state)

        assert seen == [default_store]

    def test_a_cached_loop_ignores_the_override(self, monkeypatch) -> None:
        """A second call — the idempotent get-or-create path — returns the
        already-cached loop unchanged; the override is a no-op, since only a
        first-time construction ever reads it."""
        import paramem.server.app as app_module

        cached_loop = MagicMock()
        create_calls: list = []

        def _fake_create(*args, **kwargs):
            create_calls.append((args, kwargs))
            return MagicMock()

        state = {"consolidation_loop": cached_loop}
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "create_consolidation_loop", _fake_create)

        loop = app_module.get_or_create_consolidation_loop(state, store=object())

        assert loop is cached_loop
        assert create_calls == []

    def test_run_stage_b_cycle_threads_its_store_kwarg_through(self, monkeypatch) -> None:
        """``_run_stage_b_cycle``'s own ``store=`` kwarg reaches
        ``get_or_create_consolidation_loop`` unchanged — the seam the
        weights-venue pending-event resume uses
        (``_run_pending_event_resume`` -> ``_run_stage_b_cycle``)."""
        import paramem.server.app as app_module

        sentinel_store = object()
        seen_stores: list = []

        def _fake_get_or_create(state, *, store=None):
            seen_stores.append(store)
            loop = MagicMock()
            loop._bg_trainer = None
            return loop

        state = {
            "config": MagicMock(),
        }
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "get_or_create_consolidation_loop", _fake_get_or_create)
        monkeypatch.setattr(app_module, "_active_bg_trainer", lambda config: MagicMock())

        def _body(loop, bt):
            return "noop", None

        app_module._run_stage_b_cycle(
            kind="training_crash",
            incident_key="interim",
            failure_summary="unused",
            failure_detail={},
            body=_body,
            store=sentinel_store,
        )

        assert seen_stores == [sentinel_store]

    def test_run_stage_b_cycle_default_store_is_none(self, monkeypatch) -> None:
        """The three ordinary Stage-B entry points never pass ``store=`` —
        the default ``None`` reaches ``get_or_create_consolidation_loop``
        unchanged, preserving today's behaviour."""
        import paramem.server.app as app_module

        seen_stores: list = []

        def _fake_get_or_create(state, *, store=None):
            seen_stores.append(store)
            loop = MagicMock()
            loop._bg_trainer = None
            return loop

        state = {
            "config": MagicMock(),
        }
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "get_or_create_consolidation_loop", _fake_get_or_create)
        monkeypatch.setattr(app_module, "_active_bg_trainer", lambda config: MagicMock())

        def _body(loop, bt):
            return "noop", None

        app_module._run_stage_b_cycle(
            kind="training_crash",
            incident_key="interim",
            failure_summary="unused",
            failure_detail={},
            body=_body,
        )

        assert seen_stores == [None]


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

    def test_base_swap_active_returns_deferred_string(self, monkeypatch) -> None:
        """migration.base_swap_active=True → ``"deferred_base_swap_active"``,
        the same ``deferred_*`` string shape every other arm here returns.
        This is a belt guard, not the primary refusal: on every production
        path ``base_swap_active`` is set only after ``migration["state"]`` is
        already ``"TRIAL"``, so ``_trial_active()`` (checked further down)
        already refuses in practice — this arm exists in case the flag ever
        lags the state.  All six callers of this guard (the four
        consolidation routes via ``_dispatch_consolidation``, plus
        ``/speaker/forget`` and ``/interim/discard``, which call this
        function directly) turn the string into a 409 via their own verdict
        maps.
        """
        import paramem.server.app as app_module

        state = _make_dispatch_state()
        state["migration"] = {"base_swap_active": True}
        monkeypatch.setattr(app_module, "_state", state)

        assert app_module._consolidation_dispatch_guards() == "deferred_base_swap_active"

    def test_base_swap_active_is_checked_before_the_other_guards(self, monkeypatch) -> None:
        """base_swap_active=True with consolidating=True too still returns the
        base-swap deferral, not ``deferred_already_running`` — it is the
        first check in the function.
        """
        import paramem.server.app as app_module

        state = _make_dispatch_state(consolidating=True)
        state["migration"] = {"base_swap_active": True}
        monkeypatch.setattr(app_module, "_state", state)

        assert app_module._consolidation_dispatch_guards() == "deferred_base_swap_active"

    def test_base_swap_inactive_does_not_defer(self, monkeypatch) -> None:
        """migration present but base_swap_active=False (or absent) is not a
        refusal on its own — falls through to the other guards / None."""
        import paramem.server.app as app_module

        state = _make_dispatch_state()
        state["migration"] = {"base_swap_active": False}
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() is None


# ---------------------------------------------------------------------------
# TestActiveConsolidationPendingRecordVerdict — the pending-record arm, a
# distinct verdict from the five in-flight busy arms.
# ---------------------------------------------------------------------------


class TestActiveConsolidationPendingRecordVerdict:
    """A pending stage ledger with ``consolidating`` clear is a distinct
    verdict (``deferred_event_pending``) from every in-flight busy arm, and
    ``refusal_for`` maps it to its own error code (``consolidation_pending``)
    rather than any of the five busy-arm codes."""

    def test_no_pending_ledger_and_no_busy_guard_is_clear(self, tmp_path, monkeypatch) -> None:
        import paramem.server.app as app_module

        state = _make_dispatch_state(tmp_path=tmp_path)
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module.active_consolidation() is None

    def test_pending_ledger_with_consolidating_clear_is_a_distinct_verdict(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_dispatch_state(tmp_path=tmp_path, consolidating=False)
        monkeypatch.setattr(app_module, "_state", state)

        assert app_module.active_consolidation() == "deferred_event_pending"

    def test_busy_arm_answers_before_the_pending_record_even_when_both_hold(
        self, tmp_path, monkeypatch
    ) -> None:
        """A running event past phase 1 holds both arms — ``consolidating``
        was set at dispatch and its ledger already exists on disk — and the
        busy arm answers first."""
        import paramem.server.app as app_module

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_dispatch_state(tmp_path=tmp_path, consolidating=True)
        monkeypatch.setattr(app_module, "_state", state)

        assert app_module.active_consolidation() == "deferred_already_running"

    @pytest.mark.parametrize(
        "event,expected_action",
        [
            ("interim", "interim"),
            ("full", "full"),
            ("reconcile", "reconcile"),
        ],
    )
    def test_refusal_for_names_the_pending_action(
        self, tmp_path, monkeypatch, event, expected_action
    ) -> None:
        import paramem.server.app as app_module

        _write_pending_ledger(tmp_path, event=event)
        state = _make_dispatch_state(tmp_path=tmp_path, consolidating=False)
        monkeypatch.setattr(app_module, "_state", state)

        verdict = app_module.active_consolidation()
        assert verdict == "deferred_event_pending"

        error, message = app_module.refusal_for(
            verdict, doing="forgetting a speaker", then="forget"
        )
        assert error == "consolidation_pending"
        assert error not in {
            "already_running",
            "cloud_only",
            "bg_training",
            "trial_active",
            "base_swap_active",
        }
        assert expected_action in message
        assert "before forgetting a speaker" in message
        # The pending record clears by finishing the run, or is superseded
        # by restoring a healthy backup when it keeps failing to resume.
        assert "POST /consolidate" in message
        assert "POST /backup/restore" in message

    def test_refusal_for_pending_record_message_differs_only_in_the_doing_clause(
        self, tmp_path, monkeypatch
    ) -> None:
        """Two doors sharing the same pending-record verdict differ only in
        the ``doing`` clause the caller supplies — same error code, same
        clearing-mechanism prose."""
        import paramem.server.app as app_module

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_dispatch_state(tmp_path=tmp_path, consolidating=False)
        monkeypatch.setattr(app_module, "_state", state)

        error1, message1 = app_module.refusal_for(
            "deferred_event_pending", doing="forgetting a speaker", then="forget"
        )
        error2, message2 = app_module.refusal_for(
            "deferred_event_pending", doing="discarding the interim ring", then="discard"
        )
        assert error1 == error2 == "consolidation_pending"
        assert "before forgetting a speaker" in message1
        assert "before discarding the interim ring" in message2
        # Same clearing-mechanism prose in both.
        common_tail = message1.split("wait before", 1)[0]
        assert message2.startswith(common_tail)


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
    store=None,
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
        store: Seeds ``_state["memory_store"]`` — the ``MemoryStore`` the
            content gate reads for ``RECONCILE``.  ``None`` (the default)
            preserves today's fixtures (no ``memory_store`` key at all, the
            pre-boot/test posture the gate treats as unprovable).
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

    buffer = SessionBuffer(tmp_path / "sessions", debug=False)
    for i in range(named_sessions):
        buffer.append(f"conv-named-{i}", "user", "Hello", speaker_id=f"speaker{i + 1}")
        buffer.append(f"conv-named-{i}", "assistant", "Hi")
    for i in range(anon_sessions):
        buffer.append(f"conv-anon-{i}", "user", "Hello")
        buffer.append(f"conv-anon-{i}", "assistant", "Hi")

    speaker_store = MagicMock()
    speaker_store.is_anonymous.return_value = False

    state = {
        "config": cfg,
        "session_buffer": buffer,
        "speaker_store": speaker_store,
        "consolidating": False,
        "mode": "local",
        "background_trainer": None,
        "cloud_only_reason": None,
        "last_chat_monotonic": None,
        "pending_rehydration": False,
        "integrity_check_failed": False,
    }
    if store is not None:
        state["memory_store"] = store
    return state


def _make_interim_slot(adapter_dir, stamp: str, *, payload: str | None) -> None:
    """Create ``episodic/interim_<stamp>/`` with (or without) a venue payload.

    Both venues now write into a timestamped slot SUBDIRECTORY carrying its
    own ``meta.json`` (the uniform slot-candidate shape ``count_slot_candidates``
    checks) -- a bare payload file at the interim dir root, or a payload
    directory with no ``meta.json``, is invisible to the content gate.

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
        slot = d / f"{stamp}-slot"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "meta.json").write_text("{}")
        (slot / "graph.json").write_text("{}")
    elif payload == "weights":
        slot = d / f"{stamp}-slot"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "meta.json").write_text("{}")
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


def _submitted_full_fold_events(spy) -> list[str]:
    """The ``event`` door name each submitted full-topology fold was bound to.

    The arbitrator submits ``functools.partial(_run_full_consolidation_sync,
    event)``, so the door name it chose for the fold is readable off the
    partial's bound arguments — ``"full"`` for a FULL dispatch, ``"reconcile"``
    for a RECONCILE one, both below the arbitrator running the identical fold
    topology.
    """
    import paramem.server.app as app_module

    sources = []
    for fn in spy.submitted:
        assert getattr(fn, "func", None) is app_module._run_full_consolidation_sync, (
            f"not a full-fold submission: {fn!r}"
        )
        sources.append(fn.args[0])
    return sources


class TestTierUnverifiedDeferral:
    """A main tier's registry↔slot-manifest binding failing verification --
    ``adapter_manifest_status[tier]["status"] in BINDING_ROW_STATUSES`` --
    defers every consolidation action, before the idle debounce and the
    content gate. Distinct from the store-quarantine check
    (``tests/server/test_store_quarantine.py``): this is drift a fold's own
    post-cycle revalidation observed AFTER the last successful boot/lift
    store step, not (yet) caught by a fresh one."""

    @pytest.mark.parametrize("action_name", ["AUTO", "FULL", "INTERIM", "RECONCILE"])
    @pytest.mark.parametrize(
        "row_status",
        [
            "no_matching_slot",
            "keys_without_slot",
            "payload_mismatch",
            "key_count_mismatch",
            "registry_unverified",
        ],
    )
    def test_a_withheld_main_tier_defers_every_consolidation_action(
        self, monkeypatch, action_name, row_status
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_dispatch_state()
        state["adapter_manifest_status"] = {"episodic": {"status": row_status}}
        monkeypatch.setattr(app_module, "_state", state)

        status, resolved = app_module._dispatch_consolidation(
            getattr(ConsolidationAction, action_name)
        )

        assert status == "deferred_tier_unverified"
        assert resolved is getattr(ConsolidationAction, action_name)

    def test_a_healthy_manifest_status_does_not_defer(self, monkeypatch) -> None:
        """An empty adapter_manifest_status (every tier VERIFIED or
        NO_CANDIDATES) never trips this arm -- proven directly against the
        guard function so this pin does not depend on which later gate
        answers next."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_dispatch_state()
        state["adapter_manifest_status"] = {}
        monkeypatch.setattr(app_module, "_state", state)

        status, _resolved = app_module._dispatch_consolidation(ConsolidationAction.FULL)

        assert status != "deferred_tier_unverified"


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
        assert _submitted_full_fold_events(spy) == ["full"]
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
        assert _submitted_full_fold_events(spy) == ["full"]
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
        assert _submitted_full_fold_events(spy) == ["full"]
        assert due_calls == []

    def test_full_door_at_count_zero_dispatches_on_pending_named_session_alone(
        self, tmp_path, monkeypatch
    ) -> None:
        """FULL at max_interim_count==0, no interim slots, one NAMED session → dispatches.

        At this count no interim tier exists at all, so the fold's own
        content is the pending session it will consume directly (the
        consume-pending pre-stage inside ``_run_full_consolidation_sync``
        extracts it and threads the take as ``pending`` into
        ``loop.consolidate``).
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0, named_sessions=1)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_events(spy) == ["full"]
        assert due_calls == []

    def test_full_door_at_count_zero_absorbs_a_leftover_interim_slot(
        self, tmp_path, monkeypatch
    ) -> None:
        """FULL at max_interim_count==0 with a leftover payload-bearing interim slot dispatches.

        Simulates an operator lowering ``max_interim_count`` from >0 to 0
        after a slot was already minted: the slot is still on disk, still
        payload-bearing, and must not be stranded.  The interim-slot check
        runs unconditionally (not gated on the CURRENT count), so it is
        absorbed and reaped (every full-topology fold absorbs its interim
        ring unconditionally) even though no pending session exists at all.
        """
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=0)
        _make_interim_slot(state["config"].adapter_dir, "20260101T0000", payload="weights")

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.FULL
        assert _submitted_full_fold_events(spy) == ["full"]
        assert due_calls == []

    def test_reconcile_noops_on_an_empty_store(self, tmp_path, monkeypatch) -> None:
        """RECONCILE reaches the content gate too — a store with no active key noops.

        Also pins the gate's read-only contract: a fresh store has no
        registry for any tier at all (``tiers_with_registry() == []``), and
        the noop verdict must not mint one — guards against a future edit
        that reads the gate's answer through ``MemoryStore.registry()``
        (which MINTS an empty registry as a side effect) instead of the
        non-mutating ``active_keys_in_tier()``.
        """
        from paramem.memory.store import MemoryStore
        from paramem.server.app import ConsolidationAction

        fresh_store = MemoryStore()
        assert fresh_store.tiers_with_registry() == [], "fixture sanity: a fresh store has none"

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, store=fresh_store)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "noop_no_stored_keys"
        assert resolved is ConsolidationAction.RECONCILE
        assert spy.call_count == 0
        assert due_calls == []
        assert fresh_store.tiers_with_registry() == [], (
            "the gate must not mint a registry while answering the noop question"
        )

    def test_reconcile_dispatches_with_no_store_resident(self, tmp_path, monkeypatch) -> None:
        """No live ``memory_store`` (pre-boot/test posture) is unprovable — it dispatches."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        assert "memory_store" not in state, "fixture sanity: no store seeded"

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE
        assert due_calls == []

    def test_reconcile_dispatches_when_a_main_tier_holds_one_active_key(
        self, tmp_path, monkeypatch
    ) -> None:
        """One active key in a main tier is enough content to dispatch."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import ConsolidationAction

        store = MemoryStore()
        store.put(
            "semantic", "graph1", {"key": "graph1", "subject": "a", "predicate": "b", "object": "c"}
        )

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, store=store)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE
        assert due_calls == []

    def test_reconcile_dispatches_when_only_an_interim_tier_holds_an_active_key(
        self, tmp_path, monkeypatch
    ) -> None:
        """A RECONCILE is a full consolidation whose input excludes pending
        sessions: it recalls and absorbs the interim ring exactly like any
        full fold, so an active key living ONLY in an interim tier -- no
        main tier holds one -- is still content, not a noop."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import ConsolidationAction

        store = MemoryStore()
        store.put(
            "episodic_interim_20260101T0000",
            "graph1",
            {"key": "graph1", "subject": "a", "predicate": "b", "object": "c"},
        )

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, store=store)
        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE
        assert due_calls == []

    def test_reconcile_noop_does_not_move_the_cadence_stamp(self, tmp_path, monkeypatch) -> None:
        """A RECONCILE noop leaves the durable cadence-stamp file untouched.

        Not an ordering pin — ``_stamp_scheduled_run`` only ever fires for an
        ``AUTO``-originated dispatch, and ``RECONCILE`` is never resolved from
        ``AUTO``, so this holds regardless of where the content gate sits
        relative to the stamp write.  It is a direct-call-site invariant:
        ``/reconsolidate`` never moves the cadence window, noop or not.  See
        ``test_scheduled_full_or_interim_noop_does_not_move_the_cadence_stamp``
        for the actual gate-before-stamp ordering pin (an ``AUTO`` tick that
        noops).
        """
        from paramem.memory.store import MemoryStore
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, store=MemoryStore())
        state_dir = state["config"].paths.data / "state"
        seeded_stamp = time.time() - 86400
        write_last_scheduled_run(state_dir, seeded_stamp)

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "noop_no_stored_keys"
        assert resolved is ConsolidationAction.RECONCILE
        assert spy.call_count == 0
        assert read_last_scheduled_run(state_dir) == seeded_stamp

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

    @pytest.mark.parametrize(
        ("max_interim_count", "expected_resolved"),
        [(0, "FULL"), (7, "INTERIM")],
    )
    def test_scheduled_tick_noop_does_not_stamp(
        self, tmp_path, monkeypatch, max_interim_count, expected_resolved
    ) -> None:
        """The real ordering pin: an AUTO tick resolving to FULL or INTERIM,
        with nothing for the content gate to consume, must not advance the
        cadence stamp.

        Unlike a directly requested door (never stamps regardless of outcome —
        see the other tests in this class), an ``AUTO`` dispatch DOES stamp on
        a successful one (:meth:`test_scheduled_full_stamps`,
        :meth:`test_scheduled_interim_stamps`), so a stamp advancing here would
        be a genuine call-site reordering bug — the content gate must run
        BEFORE ``_stamp_scheduled_run`` inside ``_dispatch_consolidation``.
        """
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=max_interim_count, refresh_cadence="every 5h"
        )
        state_dir = state["config"].paths.data / "state"
        seeded_stamp = time.time() - 6 * 3600
        write_last_scheduled_run(state_dir, seeded_stamp)

        status, resolved, stamp_calls = self._dispatch_and_track_stamp(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status.startswith("noop_")
        assert resolved is getattr(ConsolidationAction, expected_resolved)
        assert stamp_calls == 0, "an AUTO noop must not advance the cadence window"
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

    def _record(fn, status, **kwargs):
        submitted.append((fn, status))
        return status

    state.setdefault("migration", {})
    monkeypatch.setattr(app_module, "_state", state)
    monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
    monkeypatch.setattr(app_module, "_dispatch_to_executor", _record)
    return TestClient(app_module.app, raise_server_exceptions=False), submitted


def _route_events(submitted) -> list[str]:
    """The ``event`` door name bound to each full fold that reached the executor ritual."""
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
        assert _route_events(submitted) == ["full"]

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
        assert _route_events(submitted) == ["full"]

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
        assert _route_events(submitted) == ["full"]

    def test_reconsolidate_runs_with_nothing_new_to_consume(self, tmp_path, monkeypatch) -> None:
        """Nothing on disk, nothing pending → ``/reconsolidate`` still dispatches.

        Its input is the knowledge already stored, not the interim/pending
        material the other three doors check; "nothing new" is not a reason
        to refuse it.  This is what it is for after a model/prompt/extraction
        change.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.json() == {"status": "started_full", "action": "reconcile"}
        assert len(submitted) == 1

    def test_reconsolidate_noops_on_an_empty_store(self, tmp_path, monkeypatch) -> None:
        """A resident store with no active key in any tier → ``noop_no_stored_keys``.

        Route-level pin of the empty-store outcome, distinct from "nothing
        new" above — an operator calling ``POST /reconsolidate`` before any
        fact has ever been learned gets a `noop`, not a submitted GPU cycle.
        """
        from paramem.memory.store import MemoryStore

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, store=MemoryStore())

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/reconsolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "noop_no_stored_keys", "action": "reconcile"}
        assert submitted == []


class TestCalibrateRespondRouteDoesNotDeferOnItself:
    """Regression: ``POST /calibrate/respond`` must not self-defer via the
    arbitrator's own idle debounce.

    The route used to stamp ``_state["last_chat_monotonic"]`` (the marker
    that protects a LIVE chat turn from a fold seizing the GPU seconds
    later) before dispatching itself — so the idle-debounce check always
    read an elapsed time of ~0s and answered ``deferred_idle`` on every
    call, regardless of the debounce window.  A calibration probe of the
    serving path is not a live turn; the fix is to never stamp the marker
    from this route.  Runs the REAL arbitrator (``_route_client`` stubs
    only the executor submission) against ``tests/fixtures/server.yaml``'s
    real ``consolidation.training_idle_debounce_s`` (30s) — a MagicMock
    config would risk masking the exact arithmetic the bug lived in.
    """

    def _state(self, tmp_path, monkeypatch):
        import paramem.server.app as app_module
        from paramem.server.config import load_server_config

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        cfg = load_server_config("tests/fixtures/server.yaml")
        cfg.consolidation.calibrate_endpoint_enabled = True
        state["config"] = cfg
        state["model"] = MagicMock(name="model")
        state["tokenizer"] = MagicMock(name="tokenizer")
        state["memory_store"] = MagicMock(name="memory_store")
        state["router"] = MagicMock(name="router")
        state["speaker_store"].get_name.return_value = "Alex"
        state["calibration_run"] = None
        monkeypatch.setattr(app_module, "_abort_background_training_for_inference", lambda: None)
        return state

    def test_started_calibration_when_last_chat_monotonic_is_none(self, tmp_path, monkeypatch):
        """No prior chat turn at all — the ordinary case, and the one the
        bug broke: the route's own self-stamp made even a fresh server
        defer its first /calibrate/respond call."""
        state = self._state(tmp_path, monkeypatch)
        assert state["last_chat_monotonic"] is None

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/calibrate/respond", json={"text": "Hello", "speaker_id": "speaker1"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "started_calibration"
        assert body["action"] == "calibrate"
        assert len(submitted) == 1
        # The bug's own signature: the route must not have stamped the
        # marker as a side effect of this call.
        assert state["last_chat_monotonic"] is None

    def test_still_defers_on_a_genuinely_recent_unrelated_chat_turn(self, tmp_path, monkeypatch):
        """The debounce itself is unchanged and still protects a genuinely
        recent LIVE ``/chat`` turn from any GPU-seizing dispatch, calibrate
        included — the fix is narrowly "this route does not stamp the
        marker ITSELF", not "this route is exempt from the debounce".  A
        marker set moments ago by something else (a real chat turn) must
        still defer this call."""
        state = self._state(tmp_path, monkeypatch)
        state["last_chat_monotonic"] = time.monotonic()  # a real turn, "just now"

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/calibrate/respond", json={"text": "Hello", "speaker_id": "speaker1"})

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "deferred_idle", "action": "calibrate"}
        assert submitted == []


class TestReconsolidatePendingRecordResume:
    """``/reconsolidate`` has no special relationship to a pending
    consolidation event's record: it never discards one, and it is not a
    recovery or abandon door.  A pending record is resumed and finished
    first, exactly like the other three consolidation endpoints -- the
    identical resume-pending-first contract, never a RECONCILE-only
    carve-out."""

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

    def test_reconcile_dispatches_with_no_new_material_and_no_store_resident(
        self, tmp_path, monkeypatch
    ) -> None:
        """Nothing new to consume: every door noops except ``/reconsolidate``.

        ``/scheduled-tick`` resolves ``AUTO`` to INTERIM (no interim slots at
        all, so ``_is_full_cycle_due`` is False) and noops for lack of pending
        sessions; ``/consolidate`` requests ``FULL`` directly and noops for
        lack of a content-bearing interim slot — a DIFFERENT status, because
        it is a different action with a different input.
        ``/consolidate/interim`` requests ``INTERIM`` directly and noops the
        same way the tick did.  ``/reconsolidate`` requests ``RECONCILE``, a
        full consolidation whose content is any tier's own stored active
        keys (main or interim) rather than the pending material the other
        three check — with no ``memory_store`` resident (this fixture's
        default) that question is unprovable, so the gate lets it proceed
        rather than reading it as empty.
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

    def test_all_four_routes_defer_with_200_while_a_base_swap_is_active(
        self, tmp_path, monkeypatch
    ) -> None:
        """base_swap_active=True with ``migration["state"]`` still ``"LIVE"``
        (the flag-only belt shape — never a reachable production state, since
        every production path sets ``state="TRIAL"`` before the flag; see
        ``_consolidation_dispatch_guards``'s docstring) → every one of the
        four consolidation routes takes their ordinary HTTP 200 ``deferred_*``
        convention with ``status="deferred_base_swap_active"``, and nothing
        is dispatched.  Clearing the flag lets dispatch proceed again on the
        same client/state.
        """
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")
        state["migration"] = {"base_swap_active": True}

        client, submitted = _route_client(state, monkeypatch)

        for path in ("/scheduled-tick", "/consolidate", "/consolidate/interim", "/reconsolidate"):
            resp = client.post(path)
            assert resp.status_code == 200, path
            assert resp.json()["status"] == "deferred_base_swap_active", path

        assert submitted == [], "no dispatch may occur while a base swap is active"

        state["migration"]["base_swap_active"] = False
        resp = client.post("/consolidate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started_full"


# ---------------------------------------------------------------------------
# TestFullConsolidationFoldEntry
# ---------------------------------------------------------------------------


def _run_sync(state: dict, monkeypatch, event: str = "full") -> None:
    """Run _run_full_consolidation_sync with an inlined BackgroundTrainer."""
    import paramem.server.app as app_module

    monkeypatch.setattr(app_module, "_state", state)
    mock_bt = MagicMock()
    # submit() calls the closure synchronously so state can be inspected after.
    mock_bt.submit.side_effect = lambda fn, **kw: fn()

    with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
        app_module._run_full_consolidation_sync(event)


class TestFullConsolidationFoldEntry:
    """_run_full_consolidation_sync drives the fold entry with its venue, its
    resolved door name (event), and fold inputs."""

    def test_simulate_mode_uses_the_same_entry(self, monkeypatch, tmp_path) -> None:
        """Simulate mode routes through the identical call — only ``mode`` differs."""
        state = _make_dispatch_state(consolidation_mode="simulate", tmp_path=tmp_path)

        with patch("paramem.server.app._revalidate_adapter_manifests"):
            _run_sync(state, monkeypatch)

        loop = state["consolidation_loop"]
        loop.consolidate.assert_called_once()
        _, kwargs = loop.consolidate.call_args
        assert kwargs["mode"] == "simulate"
        assert kwargs["pending"] is None

    def test_empty_tiers_rebuilt_is_a_noop_terminal(self, monkeypatch, tmp_path) -> None:
        """tiers_rebuilt == [] ends the cycle as a noop for every caller.

        The flag that used to exempt the on-demand fold from this guard is gone:
        an empty rebuild is a noop no matter who dispatched it, and the
        ``consolidating`` flag is cleared on the way out.
        """
        state = _make_dispatch_state(tmp_path=tmp_path)
        state["consolidating"] = True  # set by the dispatcher before submit

        with patch("paramem.server.app._revalidate_adapter_manifests"):
            _run_sync(state, monkeypatch)

        assert state["consolidating"] is False, (
            "_state['consolidating'] must be cleared after the fold completes"
        )

    def test_reconcile_event_never_consumes_pending_sessions(self, monkeypatch, tmp_path) -> None:
        """A reconcile event is a full consolidation whose input excludes
        pending sessions: at ``max_interim_count == 0`` (where an ordinary
        full fold's pre-stage would extract pending sessions directly) a
        reconcile event still runs no extraction pre-stage and passes
        ``pending=None`` to ``loop.consolidate`` -- pending sessions stay
        pending."""
        import paramem.server.app as app_module

        state = _make_dispatch_state(
            consolidation_mode="train", max_interim_count=0, tmp_path=tmp_path
        )
        extract_spy = MagicMock()
        monkeypatch.setattr(app_module, "_extract_pending_sessions", extract_spy)

        with patch("paramem.server.app._revalidate_adapter_manifests"):
            _run_sync(state, monkeypatch, event="reconcile")

        loop = state["consolidation_loop"]
        loop.consolidate.assert_called_once()
        _, kwargs = loop.consolidate.call_args
        assert kwargs["event"] == "reconcile"
        assert kwargs["pending"] is None
        extract_spy.assert_not_called()

    def test_full_trained_run_status_detail_has_no_extra_fields(
        self, monkeypatch, tmp_path
    ) -> None:
        """``_finalize_full`` records exactly this detail key set.

        A rebuilt tier drives the cycle to the ``full_trained`` terminal,
        which persists a run-status record via ``record_last_run``. The
        recorded detail must carry exactly ``tiers_rebuilt`` and
        ``total_keys`` — no more, no fewer — so a writer recording any extra
        field fails this pin even though the renderer would tolerate it
        silently.
        """
        from paramem.server.run_status import read_last_runs

        state = _make_dispatch_state(
            consolidation_mode="train",
            tmp_path=tmp_path,
            consolidate_return={
                "tiers_rebuilt": ["episodic"],
                "completed": True,
                "aborted": False,
            },
        )

        with patch("paramem.server.app._revalidate_adapter_manifests"):
            _run_sync(state, monkeypatch)

        last_runs = read_last_runs(tmp_path / "state")
        record = last_runs["consolidation"]
        assert record.outcome == "full_trained"
        assert set(record.detail) == {"tiers_rebuilt", "total_keys"}


class TestFinalizeFullAbortedResume:
    """``_finalize_full`` must not report success or clear crash incidents
    for an aborted/incomplete resumed-full result (``result["aborted"]`` or
    ``result["completed"] is False``) -- the shape ``_finish_resumed_event``
    produces when a resume's bundle yields mid-publish."""

    def _aborted_result(self) -> dict:
        return {
            "tiers_rebuilt": [],
            "consumed_session_ids": [],
            "consumed_episodic_rels": 0,
            "consumed_procedural_rels": 0,
            "completed": False,
            "aborted": True,
        }

    def test_aborted_resume_records_aborted_not_full_trained(self, monkeypatch, tmp_path) -> None:
        import paramem.server.app as app_module
        from paramem.server.run_status import read_last_runs

        state = _make_dispatch_state(tmp_path=tmp_path)
        monkeypatch.setattr(app_module, "_state", state)
        loop = MagicMock()
        loop.store.all_active_keys.return_value = []

        app_module._finalize_full(loop, self._aborted_result())

        record = read_last_runs(tmp_path / "state")["consolidation"]
        assert record.outcome == "aborted"

    def test_aborted_resume_does_not_stamp_last_consolidation(self, monkeypatch, tmp_path) -> None:
        import paramem.server.app as app_module

        state = _make_dispatch_state(tmp_path=tmp_path)
        state["last_consolidation"] = None
        monkeypatch.setattr(app_module, "_state", state)
        loop = MagicMock()
        loop.store.all_active_keys.return_value = []

        app_module._finalize_full(loop, self._aborted_result())

        assert state["last_consolidation"] is None
        assert state["consolidating"] is False

    def test_aborted_resume_leaves_consolidation_crash_incident_unresolved(
        self, monkeypatch, tmp_path
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.incidents import read_incidents, record_incident

        state = _make_dispatch_state(tmp_path=tmp_path)
        monkeypatch.setattr(app_module, "_state", state)
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="consolidation_crash",
            key="full",
            severity="failed",
            summary="Full consolidation crashed unexpectedly",
            detail={},
        )
        loop = MagicMock()
        loop.store.all_active_keys.return_value = []

        app_module._finalize_full(loop, self._aborted_result())

        active = [inc for inc in read_incidents(state_dir) if inc.status == "active"]
        assert any(inc.type == "consolidation_crash" for inc in active), (
            "an aborted resumed full event must not clear a pre-recorded "
            "consolidation_crash incident"
        )

    def test_completed_full_still_records_full_trained_and_resolves(
        self, monkeypatch, tmp_path
    ) -> None:
        """Control: a completed (non-aborted) result keeps the pre-fix
        behavior -- outcome ``full_trained``, ``last_consolidation`` stamped,
        incidents resolved."""
        import paramem.server.app as app_module
        from paramem.server.incidents import read_incidents, record_incident
        from paramem.server.run_status import read_last_runs

        state = _make_dispatch_state(tmp_path=tmp_path)
        state["last_consolidation"] = None
        monkeypatch.setattr(app_module, "_state", state)
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="consolidation_crash",
            key="full",
            severity="failed",
            summary="Full consolidation crashed unexpectedly",
            detail={},
        )
        loop = MagicMock()
        loop.store.all_active_keys.return_value = ["k1"]

        completed_result = {
            "tiers_rebuilt": ["episodic"],
            "consumed_session_ids": [],
            "consumed_episodic_rels": 0,
            "consumed_procedural_rels": 0,
            "completed": True,
            "aborted": False,
        }
        app_module._finalize_full(loop, completed_result)

        record = read_last_runs(state_dir)["consolidation"]
        assert record.outcome == "full_trained"
        assert state["last_consolidation"] is not None
        active = [inc for inc in read_incidents(state_dir) if inc.status == "active"]
        assert not any(inc.type == "consolidation_crash" for inc in active)


class TestFinalizeInterimDetailCarriesProceduralCount:
    """``_finalize_interim``'s durable run-status detail reports the event's
    procedural-relation count from ``result["consumed_procedural_rels"]``
    (the ledger's own extraction stage) — a cycle that captured preference-
    typed relations must not record ``procedural_rels: 0`` in the operator-
    visible run record."""

    def test_nonzero_procedural_count_reaches_the_run_status_detail(
        self, monkeypatch, tmp_path
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.run_status import read_last_runs

        state = _make_dispatch_state(tmp_path=tmp_path)
        monkeypatch.setattr(app_module, "_state", state)
        state_dir = tmp_path / "state"

        loop = MagicMock()
        loop.store.all_active_keys.return_value = ["k1", "k2"]
        loop._fold_state_dir = state_dir

        result = {
            "tiers_rebuilt": ["episodic_interim_20260101T0000"],
            "consumed_session_ids": ["s1"],
            "consumed_episodic_rels": 2,
            "consumed_procedural_rels": 3,
            "completed": True,
            "aborted": False,
            "adapter_name": "episodic_interim_20260101T0000",
            "mode": "trained",
            "new_keys": [],
            "triples_extracted": 5,
        }
        app_module._finalize_interim(loop, result)

        record = read_last_runs(state_dir)["consolidation"]
        assert record.detail["procedural_rels"] == 3, (
            f"the run-status detail must carry the event's own procedural count; "
            f"got {record.detail}"
        )
        assert record.detail["episodic_rels"] == 2


class TestFinalizeInterimAbortedGating:
    """``_finalize_interim``'s ``training_crash``/``vram_exhausted``
    auto-resolve must gate on ``result["completed"]`` -- this finalizer is
    the terminal for every non-crash interim outcome (``trained`` /
    ``simulated`` / ``cap_pending`` / ``aborted``), not only a clean
    success."""

    def test_aborted_interim_leaves_training_crash_incident_unresolved(
        self, monkeypatch, tmp_path
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.incidents import read_incidents, record_incident

        state = _make_dispatch_state(tmp_path=tmp_path)
        monkeypatch.setattr(app_module, "_state", state)
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="training_crash",
            key="interim",
            severity="failed",
            summary="Interim training crashed unexpectedly",
            detail={},
        )
        loop = MagicMock()
        loop.store.all_active_keys.return_value = []
        loop._fold_state_dir = state_dir

        result = {
            "tiers_rebuilt": [],
            "consumed_session_ids": [],
            "consumed_episodic_rels": 0,
            "consumed_procedural_rels": 0,
            "completed": False,
            "aborted": True,
            "adapter_name": "episodic_interim_20260101T0000",
            "mode": "aborted",
            "new_keys": [],
            "triples_extracted": 0,
        }
        app_module._finalize_interim(loop, result)

        active = [inc for inc in read_incidents(state_dir) if inc.status == "active"]
        assert any(inc.type == "training_crash" for inc in active), (
            "an aborted interim cycle must not clear a pre-recorded training_crash incident"
        )

    def test_completed_interim_still_resolves_training_crash(self, monkeypatch, tmp_path) -> None:
        """Control: a completed interim result keeps the pre-fix behavior."""
        import paramem.server.app as app_module
        from paramem.server.incidents import read_incidents, record_incident

        state = _make_dispatch_state(tmp_path=tmp_path)
        monkeypatch.setattr(app_module, "_state", state)
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="training_crash",
            key="interim",
            severity="failed",
            summary="Interim training crashed unexpectedly",
            detail={},
        )
        loop = MagicMock()
        loop.store.all_active_keys.return_value = ["k1"]
        loop._fold_state_dir = state_dir

        result = {
            "tiers_rebuilt": ["episodic_interim_20260101T0000"],
            "consumed_session_ids": [],
            "consumed_episodic_rels": 0,
            "consumed_procedural_rels": 0,
            "completed": True,
            "aborted": False,
            "adapter_name": "episodic_interim_20260101T0000",
            "mode": "trained",
            "new_keys": [],
            "triples_extracted": 0,
        }
        app_module._finalize_interim(loop, result)

        active = [inc for inc in read_incidents(state_dir) if inc.status == "active"]
        assert not any(inc.type == "training_crash" for inc in active)


# ---------------------------------------------------------------------------
# TestResumeArbitrationMatrix — the resume-pending-first arm of
# _dispatch_consolidation / _dispatch_resume.  Every dispatch that finds a
# pending ledger resumes and finishes THAT event before any new one starts,
# regardless of which action was requested; the resumed action is read off
# the ledger head, never the request.  _dispatch_resume itself carried zero
# test references before this class (verified: only _run_pending_event_resume
# -- reached through it -- was exercised, by tests/test_fold_crash_resume.py,
# and only by calling it directly, never through the arbitrator).
# ---------------------------------------------------------------------------


class TestResumeArbitrationMatrix:
    """``_dispatch_consolidation`` resumes a pending event ahead of starting
    a fresh one, for every requested action -- the no-re-staging
    invariant's sole test home.  Uses the executor spy (nothing the resume
    submits is ever actually run), so these pins are about WHICH function
    gets submitted and WHAT status/action is reported, not the resume's own
    mechanics (covered end-to-end by ``tests/test_fold_crash_resume.py``)."""

    def test_pending_full_event_resumes_ahead_of_a_direct_interim_request(
        self, tmp_path, monkeypatch
    ) -> None:
        """A pending FULL-event ledger pre-empts a direct ``/consolidate/interim``
        request: the fold that resumes is the one already on disk, not a
        fresh interim absorb."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="full")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.FULL
        assert spy.submitted == [app_module._run_pending_event_resume]

    def test_pending_reconcile_event_resumes_and_reports_reconcile(
        self, tmp_path, monkeypatch
    ) -> None:
        """A pending RECONCILE-event ledger resumes and reports action
        ``RECONCILE`` -- reporting reads the ledger head's ``event`` field
        directly, so a resumed reconcile is never folded into the generic
        ``FULL`` action a plain ``event == "full"`` comparison would."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="reconcile")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.RECONCILE
        assert spy.submitted == [app_module._run_pending_event_resume]

    def test_pending_full_event_resumes_ahead_of_auto(self, tmp_path, monkeypatch) -> None:
        """A scheduled tick over a pending FULL-event ledger resumes it
        rather than resolving its own FULL/INTERIM deadline decision.

        Resume-pending-first is unconditional, grouped with the safety
        gates ahead of the schedule's own catch-up/due-check business
        (never gated by whether a tick is due), so ``_is_full_cycle_due``
        -- AUTO's own FULL-vs-INTERIM resolution -- is never even reached:
        the reported action comes from the ledger, never from that
        resolution.
        """
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import write_last_scheduled_run

        _write_pending_ledger(tmp_path, event="full")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="every 5h")
        write_last_scheduled_run(tmp_path / "state", time.time() - 6 * 3600)  # DUE

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.FULL
        assert spy.submitted == [app_module._run_pending_event_resume]
        assert due_calls == [], (
            "resume-pending-first pre-empts AUTO's own deadline resolution entirely"
        )

    def test_pending_interim_event_resumes_ahead_of_a_direct_full_request(
        self, tmp_path, monkeypatch
    ) -> None:
        """A pending INTERIM-event ledger pre-empts a direct ``/consolidate``
        (FULL) request."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._run_pending_event_resume]

    def test_pending_record_resumes_ahead_of_an_armed_active_store_migration(
        self, tmp_path, monkeypatch
    ) -> None:
        """A dispatch that finds BOTH a pending consolidation ledger AND an
        armed mode switch (``pending_rehydration=True``) resumes the ledger
        and reports that run -- the migration pre-empt is never reached.
        The store migration is content-preserving and needs a coherent,
        record-free tree, so it only ever runs on a LATER dispatch, once
        the pending event has resumed to completion."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["pending_rehydration"] = True

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._run_pending_event_resume], (
            "the migration pre-empt must not run while a consolidation record is still pending"
        )

    def test_not_due_auto_tick_still_resumes_a_pending_record(self, tmp_path, monkeypatch) -> None:
        """The resume arm is checked BEFORE the catch-up gate on a scheduled
        tick: a tick still inside its cadence window still resumes a
        pending ledger -- resume-pending-first is unconditional, grouped
        with the safety gates ahead of the schedule's own due-check
        business, because a pending event is in-flight state that must
        finish before an explicit request or a scheduled tick walks past
        it.  The schedule's dueness governs starting NEW work, never
        finishing what is already in flight."""
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_grammar import scheduled_run_stamp_value
        from paramem.server.schedule_state import write_last_scheduled_run

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="12h")
        write_last_scheduled_run(tmp_path / "state", scheduled_run_stamp_value("12h", time.time()))

        status, resolved, spy, due_calls = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.call_count == 1, "a not-due tick must still resume the pending event"
        assert due_calls == [], "the resume pre-empts AUTO's own due-check entirely"

    def test_max_interim_count_zero_still_resumes_a_pending_interim_record(
        self, tmp_path, monkeypatch
    ) -> None:
        """A pending INTERIM ledger resumes even at ``max_interim_count=0``
        (no interim tier exists any more) -- the resume-pending-first arm
        sits ahead of the N==0 ``noop_no_interim_tier`` tier check, so an
        operator who lowered the count mid-event still gets the in-flight
        event finished rather than a meaningless tier-not-found refusal."""
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=0)

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._run_pending_event_resume]

    def test_auto_resume_stamps_the_cadence_so_a_later_record_free_tick_reads_not_due(
        self, tmp_path, monkeypatch
    ) -> None:
        """A scheduled tick that resumes a pending event still consumes its
        cadence window.  While the record stays pending, resume-pending-
        first is unconditional and ignores the stamp entirely -- the very
        next tick resumes it again regardless of the freshly stamped
        window, since the schedule's dueness governs starting NEW work,
        never finishing what is already in flight.  Only once the event
        actually finishes (its record disposed) does a later tick inside
        the same stamped window read NOT_DUE off the stamp the resumed
        tick advanced, exactly like any other scheduled tick -- proving
        the stamp still does its ordinary job once nothing is pending."""
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run
        from paramem.training import stage_ledger as sl

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="every 5h")
        state_dir = tmp_path / "state"
        seeded_stamp = time.time() - 6 * 3600  # due
        write_last_scheduled_run(state_dir, seeded_stamp)

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )
        assert status == "started_resume"
        assert spy.call_count == 1
        stamped = read_last_scheduled_run(state_dir)
        assert stamped != seeded_stamp, "the resumed tick must advance the cadence stamp"

        # The resumed job (never actually run here -- the executor is a
        # spy) completes and clears the busy flag, the way the real
        # executor's finalizer eventually would; isolates the cadence-stamp
        # behaviour under test from the unrelated "already running" guard.
        state["consolidating"] = False

        # The record is STILL pending (the spy never actually disposed it)
        # -- the very next tick resumes it again, unconditionally, inside
        # the freshly stamped window.
        status2, _resolved2, spy2, due_calls2 = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )
        assert status2 == "started_resume"
        assert spy2.call_count == 1
        assert due_calls2 == []

        # Now the event actually finishes -- record disposed.  A later
        # tick inside the same stamped window reads NOT_DUE, same as any
        # other scheduled tick with nothing pending.
        sl.dispose(state_dir)
        state["consolidating"] = False
        status3, _resolved3, spy3, due_calls3 = _dispatch(
            state, ConsolidationAction.AUTO, monkeypatch=monkeypatch
        )
        assert status3 == "noop_not_due"
        assert spy3.call_count == 0
        assert due_calls3 == []
        assert read_last_scheduled_run(state_dir) == stamped

    @pytest.mark.parametrize(
        "event,requested,expected_resolved",
        [
            ("interim", "FULL", "INTERIM"),
            ("full", "INTERIM", "FULL"),
            ("reconcile", "INTERIM", "RECONCILE"),
        ],
    )
    def test_resumed_action_name_derives_from_the_ledger_head_not_the_request(
        self, tmp_path, monkeypatch, event, requested, expected_resolved
    ) -> None:
        """The resolved action ``_dispatch_consolidation`` returns for a
        resume is read off the ledger head's ``event`` field directly --
        never the action the caller actually requested, which the docstring
        says "waits for the next tick"."""
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event=event)
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        status, resolved, _spy, _ = _dispatch(
            state, getattr(ConsolidationAction, requested), monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is getattr(ConsolidationAction, expected_resolved)

    def test_resume_leaves_exactly_one_ledger_untouched_on_disk(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dispatching over a pending record never mints a second ledger --
        the file on disk is byte-identical before and after the arbitrator
        hands the resume to the executor (the executor itself is a spy here
        and never actually runs the resume job, so this isolates the
        arbitrator's own behaviour from the resume's)."""
        from paramem.server.app import ConsolidationAction
        from paramem.training import stage_ledger as sl

        _write_pending_ledger(tmp_path, event="full")
        state_dir = tmp_path / "state"
        ledger_bytes_before = sl.ledger_path(state_dir).read_bytes()
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        status, _resolved, _spy, _ = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert sl.ledger_path(state_dir).read_bytes() == ledger_bytes_before
        assert sl.ledger_path(state_dir).exists()
        # Exactly one ledger file, not a second one alongside it.
        assert list(state_dir.glob("*ledger*")) == [sl.ledger_path(state_dir)]

    def test_only_the_resume_job_is_ever_submitted_while_a_record_is_pending(
        self, tmp_path, monkeypatch
    ) -> None:
        """For EVERY action, including ``RECONCILE``, the ONLY function the
        arbitrator ever hands to the executor while a record is pending is
        ``_run_pending_event_resume`` -- never ``_extract_and_start_training``
        nor a ``functools.partial(_run_full_consolidation_sync, ...)``, which
        are the only two call sites that ever reach ``stage_event``.  This
        is the structural half of the no-re-staging invariant: a fresh
        staging pass is provably unreachable through any door while a
        record is pending, because the function that could reach it is
        never even submitted.  ``RECONCILE`` (``/reconsolidate``) has no
        carve-out here: it never discards a pending record, so it resumes
        exactly like the other three doors.
        """
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        for action in (
            ConsolidationAction.AUTO,
            ConsolidationAction.FULL,
            ConsolidationAction.INTERIM,
            ConsolidationAction.RECONCILE,
        ):
            sub = tmp_path / action.value
            _write_pending_ledger(sub, event="interim")
            state = _make_arbitrator_state(sub, max_interim_count=7, refresh_cadence="")

            _status, _resolved, spy, _ = _dispatch(state, action, monkeypatch=monkeypatch)

            assert spy.submitted == [app_module._run_pending_event_resume], (
                f"{action.value}: stage_event's only reachable callers must never be "
                f"submitted while a record is pending; got {spy.submitted!r}"
            )


# ---------------------------------------------------------------------------
# TestFiveDoorPendingRecordGuard — the pending-record verdict
# (``deferred_event_pending`` -> ``consolidation_pending``) answered
# identically by every mutating door that reads ``active_consolidation()``.
# Each door's own five BUSY arms (consolidating / bg-training / cloud-only /
# trial-active / base-swap-active) live with that door's own test file; this
# class pins the SIXTH, pending-record arm across all five doors in one
# place, since ``/admin/assign-orphans`` had no behavioural test at all
# before this (verified: only route-table/auth introspection exists in
# ``tests/server/test_require_admin.py``) and neither
# ``tests/server/test_speaker_forget.py`` nor
# ``tests/server/test_debug_erase_keys_endpoint.py`` exercised this arm
# (verified: no ``deferred_event_pending`` / ``consolidation_pending``
# reference in either file).
# ---------------------------------------------------------------------------


class TestFiveDoorPendingRecordGuard:
    def test_admin_assign_orphans_refuses_with_a_pending_record(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["speaker_store"].list_profiles.return_value = [{"id": "speaker0", "name": "Speaker0"}]
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/admin/assign-orphans")

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidation_pending"

    def test_speaker_forget_refuses_with_a_pending_record(self, tmp_path, monkeypatch) -> None:
        import paramem.server.app as app_module
        from tests.server._state_builders import (
            _make_forget_buffer as _make_buffer,
        )
        from tests.server._state_builders import (
            _make_forget_loop as _make_loop,
        )
        from tests.server._state_builders import (
            _make_forget_speaker_store as _make_speaker_store,
        )
        from tests.server._state_builders import _make_forget_state

        speaker_id = "speaker0"
        _write_pending_ledger(tmp_path / "data", event="interim")
        loop = _make_loop(speaker_id, [])
        state = _make_forget_state(
            tmp_path,
            loop=loop,
            speaker_store=_make_speaker_store(speaker_id),
            buffer=_make_buffer(speaker_id, []),
        )
        monkeypatch.setattr(app_module, "_state", state)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/speaker/forget", json={"speaker_id": speaker_id})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidation_pending"

    def test_debug_erase_keys_refuses_with_a_pending_record(self, tmp_path, monkeypatch) -> None:
        import paramem.server.app as app_module
        from tests.server._state_builders import _make_erase_state

        _write_pending_ledger(tmp_path / "data", event="interim")
        state = _make_erase_state(tmp_path)
        state["config"].debug = True
        monkeypatch.setattr(app_module, "_state", state)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/debug/erase-keys", json={"keys": ["graph1"], "confirm": True})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidation_pending"

    def test_interim_discard_refuses_with_a_pending_record(self, tmp_path, monkeypatch) -> None:
        """See also ``tests/server/test_interim_discard.py::TestGuardMatrix``,
        which owns this door's other five (busy) arms; this pin adds the
        sixth, pending-record arm alongside them there too."""
        import paramem.server.app as app_module
        from tests.server._state_builders import _make_discard_state

        _write_pending_ledger(tmp_path / "data", event="interim")
        state = _make_discard_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidation_pending"

    def test_ingest_sessions_cancel_refuses_with_a_pending_record(
        self, tmp_path, monkeypatch
    ) -> None:
        """See also ``tests/server/test_ingest_endpoint.py``, which owns this
        door's happy-path and not-found behaviour and adds the
        JSONLs-still-present variant of this same arm."""
        import paramem.server.app as app_module
        from tests.server._state_builders import _make_ingest_state

        _write_pending_ledger(tmp_path / "data", event="interim")
        state = _make_ingest_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/ingest-sessions/cancel", json={"session_ids": []})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidation_pending"


# ---------------------------------------------------------------------------
# TestCalibrateArbitratorStages — the arbitrator's pre-dispatch stages
# (retro-claim, retiring triage, pending-ledger resume) as they apply to the
# two calibrate actions specifically: retro-claim runs for every action,
# including calibrate; retiring triage/mark_consolidated is staging-only;
# a pending ledger makes a calibrate dispatch PROCEED rather than resume.
# ---------------------------------------------------------------------------


def _dispatch_with_retro_claim_spy(state, action, *, monkeypatch, spec=None):
    """Like ``_dispatch`` above, but spies on ``_retro_claim_orphan_sessions``
    instead of stubbing it to a no-op -- the caller needs to observe whether
    it ran, not merely tolerate it running."""
    import paramem.server.app as app_module

    calls: list[int] = []
    _real = app_module._retro_claim_orphan_sessions

    def _spy():
        calls.append(1)
        return _real()

    spy = _ExecutorSpy()
    state["event_loop"] = spy.loop
    monkeypatch.setattr(app_module, "_state", state)
    monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", _spy)
    status, resolved = app_module._dispatch_consolidation(action, spec=spec)
    return status, resolved, spy, calls


class TestRetroClaimRunsForEveryAction:
    """the retro-claim runs for every action, including both
    calibrate actions."""

    def test_retro_claim_runs_for_calibrate_pending_even_on_a_noop(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.consolidation_action import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)  # nothing pending -> noop
        status, _resolved, _spy, calls = _dispatch_with_retro_claim_spy(
            state, ConsolidationAction.CALIBRATE_PENDING, monkeypatch=monkeypatch
        )

        assert status == "noop_no_pending"
        assert calls == [1], "retro-claim must run even though this dispatch ends in a noop"

    def test_retro_claim_runs_for_calibrate(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from paramem.server.consolidation_action import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        status, _resolved, spy, calls = _dispatch_with_retro_claim_spy(
            state, ConsolidationAction.CALIBRATE, monkeypatch=monkeypatch, spec=MagicMock()
        )

        assert status == "started_calibration"
        assert calls == [1]
        assert spy.call_count == 1

    def test_retro_claim_runs_for_a_staging_action_too(self, tmp_path, monkeypatch) -> None:
        """Control: retro-claim already ran for staging actions before this
        change; still true."""
        from paramem.server.consolidation_action import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=1)
        status, _resolved, _spy, calls = _dispatch_with_retro_claim_spy(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "started"
        assert calls == [1]


class TestRetiringTriageIsStagingOnly:
    """retiring triage (and therefore ``mark_consolidated``) runs
    only for staging actions -- a calibrate dispatch classifies pending
    sessions (the counts feed the content gate) but never retires any of
    them."""

    def test_calibrate_pending_dispatch_never_calls_mark_consolidated(
        self, tmp_path, monkeypatch
    ) -> None:
        from unittest.mock import patch

        from paramem.server.consolidation_action import ConsolidationAction

        # One UNIDENTIFIABLE session (no speaker_id, no voice embedding) --
        # a staging dispatch would retire it via retire_unattributable_sessions.
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, anon_sessions=1)
        buffer = state["session_buffer"]

        with patch.object(buffer, "mark_consolidated", wraps=buffer.mark_consolidated) as spy:
            status, _resolved, _spy, _calls = _dispatch_with_retro_claim_spy(
                state, ConsolidationAction.CALIBRATE_PENDING, monkeypatch=monkeypatch
            )

        assert status == "noop_no_named"  # the UNIDENTIFIABLE session is not NAMED
        spy.assert_not_called()

    def test_the_same_unidentifiable_session_is_retired_on_a_staging_dispatch(
        self, tmp_path, monkeypatch
    ) -> None:
        """Control: the identical seeded session IS retired when a staging
        action (here INTERIM) reaches the same triage stage -- proves the
        prior test's non-call is the calibrate/staging distinction, not an
        inert fixture."""
        from unittest.mock import patch

        from paramem.server.consolidation_action import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, anon_sessions=1)
        buffer = state["session_buffer"]

        with patch.object(buffer, "mark_consolidated", wraps=buffer.mark_consolidated) as spy:
            status, _resolved, _spy, _calls = _dispatch_with_retro_claim_spy(
                state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
            )

        assert status == "noop_no_named"
        spy.assert_called_once()


class TestPendingLedgerCalibrateProceedsStagingResumes:
    """a pending stage ledger makes a calibrate dispatch PROCEED
    past the resume step, while the four staging doors resume the pending
    event instead."""

    def test_calibrate_proceeds_past_a_pending_full_event_ledger(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.consolidation_action import ConsolidationAction

        _write_pending_ledger(tmp_path, event="full")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        status, resolved, spy, _calls = _dispatch_with_retro_claim_spy(
            state, ConsolidationAction.CALIBRATE, monkeypatch=monkeypatch, spec=MagicMock()
        )

        # CALIBRATE's content gate never noops, so a resumed dispatch and a
        # proceeding dispatch are distinguishable by their terminal status
        # AND by which executor entry point was submitted: a resume submits
        # _run_pending_event_resume; a proceeding calibrate dispatch submits
        # _run_calibration_sync.
        assert status == "started_calibration"
        assert resolved is ConsolidationAction.CALIBRATE
        assert len(spy.submitted) == 1
        submitted_fn = spy.submitted[0]
        assert submitted_fn is not app_module._run_pending_event_resume
        assert getattr(submitted_fn, "func", None) is app_module._run_calibration_sync

    def test_a_staging_door_resumes_the_same_pending_ledger(self, tmp_path, monkeypatch) -> None:
        """Control: the identical pending ledger makes a directly-requested
        FULL door resume instead of proceeding."""
        from paramem.server.consolidation_action import ConsolidationAction

        _write_pending_ledger(tmp_path, event="full")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)

        status, resolved, spy, _ = _dispatch(
            state, ConsolidationAction.FULL, monkeypatch=monkeypatch
        )

        assert status == "started_resume"
        assert resolved is ConsolidationAction.FULL
