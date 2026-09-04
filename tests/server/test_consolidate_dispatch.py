"""Tests for the server-side consolidation dispatch infrastructure.

All tests are CPU-only, no model load required.  The consolidation-loop and
BackgroundTrainer are mocked so the implementation-level dispatch paths can be
verified in isolation.

Coverage:
- ``_consolidation_dispatch_guards`` shared guard helper
- ``_dispatch_consolidation`` — the arbitrator's door wrapper: ``AUTO`` is
  requested by ``/scheduled-tick``, the boot task and the idle watch, and
  ``choose_consolidation_run`` decides from the clock, the pending head, the
  two schedule marks and the full-fold deadline which run it earns.  ``FULL``,
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

import asyncio
import contextlib
import time
from datetime import datetime
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
        # create_consolidation_loop is patched at its DEFINING module
        # (paramem.server.consolidation), not at app_module -- the
        # get-or-create it backs resolves the name via its own __globals__
        # when it runs, so a patch placed on the importing module has no
        # effect (mirrors the sibling test above,
        # TestConsolidationLoopStoreOverride's first-construction case).
        import paramem.server.consolidation as consolidation_module

        monkeypatch.setattr(consolidation_module, "create_consolidation_loop", _fake_create)

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
        unchanged."""
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
            own stamp state directly.  ``""`` is manual-only (no
            cadence, no stamp seeded) — used by the manual-only-posture
            tests, where ``FULL``/``INTERIM`` requested directly are the
            only doors that ever fire.
        period_seconds: ``config.consolidation.consolidation_period_seconds`` —
            the age at which the oldest interim slot makes a full fold due,
            read only when a scheduled dispatch computes the full-fold
            deadline.  ``None`` (the default) is a manual-only cadence: no
            deadline, so no scheduled full fold for any interim ring.  A
            directly requested ``FULL`` never reads this at all.
        store: Seeds ``_state["memory_store"]`` — the ``MemoryStore`` the
            content gate reads for ``RECONCILE``.  ``None`` (the default)
            preserves today's fixtures (no ``memory_store`` key at all, the
            pre-boot/test posture the gate treats as unprovable).
    """
    from paramem.server.schedule_grammar import parse_schedule_atom
    from paramem.server.schedule_state import ScheduleMarks, write_marks
    from paramem.server.session_buffer import SessionBuffer

    cfg = MagicMock()
    cfg.consolidation.mode = consolidation_mode
    cfg.consolidation.max_interim_count = max_interim_count
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.consolidation.interim_resume = "immediate"
    cfg.consolidation.full_window = "01:00-04:00"
    cfg.consolidation.training_idle_debounce_s = 30
    cfg.session.idle_timeout_minutes = 10
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
        write_marks(tmp_path / "state", ScheduleMarks(time.time() - 86400, None))

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
        "last_model_use_monotonic": None,
        "pending_rehydration": False,
        "integrity_check_failed": False,
    }
    if store is not None:
        state["memory_store"] = store
    return state


def _make_interim_slot(adapter_dir, stamp: str, *, payload: str | None) -> None:
    """Create ``episodic/interim_<stamp>/`` with (or without) a venue payload.

    Both venues write into a timestamped slot SUBDIRECTORY carrying its
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


def _unwrap_evicted(fn):
    """Unwrap ``_dispatch_to_executor``'s ``functools.partial(_run_evicted, fn)``
    wrapper down to the entry point it wraps.

    Every real dispatch wraps its submission this way (see
    ``_dispatch_to_executor`` in ``paramem/server/app.py``) so the executor
    evicts the GPU voice pair before the run's own entry point starts.  This
    is the ONE unwrap site in this module: ``_ExecutorSpy`` calls it at
    capture time, so every identity/bound-args assertion below reads the
    unwrapped entry point without its own unwrap logic.  Asserts the wrapper
    shape it expects, so a submission that is NOT wrapped this way fails
    loudly here rather than producing a confusing identity mismatch three
    frames away.
    """
    import paramem.server.app as app_module

    assert getattr(fn, "func", None) is app_module._run_evicted, (
        f"expected _dispatch_to_executor to wrap every submission in "
        f"functools.partial(_run_evicted, fn); got {fn!r}"
    )
    (inner,) = fn.args
    return inner


class _ExecutorSpy:
    """Stand-in for the event loop: records what was submitted, runs nothing.

    Records the UNWRAPPED entry point (:func:`_unwrap_evicted`) — the
    eviction wrapper itself is an envelope property with its own coverage
    (``tests/server/test_extraction_stage_lifecycle.py``), not something
    every arbitrator test in this module should have to see.
    """

    def __init__(self) -> None:
        self.submitted: list[object] = []
        self.loop = MagicMock()
        self.loop.run_in_executor.side_effect = self._submit

    def _submit(self, executor, fn):
        self.submitted.append(_unwrap_evicted(fn))
        future = MagicMock()
        future.add_done_callback.return_value = None
        return future

    @property
    def call_count(self) -> int:
        return len(self.submitted)


def _reason_for(action):
    """The dispatch reason a door passes for *action*: the timer for ``AUTO``,
    an operator for every other action."""
    from paramem.server.app import ConsolidationAction
    from paramem.server.consolidation_choice import DispatchReason

    return DispatchReason.TIMER if action is ConsolidationAction.AUTO else DispatchReason.OPERATOR


def _dispatch(state, action, *, monkeypatch=None):
    """Run the arbitrator against *state*, capturing executor submissions.

    Returns ``(status, resolved_action, spy)``.  The reason passed is the one
    the production door for *action* passes (see ``_reason_for``).
    """
    import paramem.server.app as app_module

    spy = _ExecutorSpy()
    state["event_loop"] = spy.loop
    monkeypatch.setattr(app_module, "_state", state)
    monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
    status, resolved = app_module._dispatch_consolidation(action, reason=_reason_for(action))
    return status, resolved, spy


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


class TestConsolidationArbitrator:
    """_dispatch_consolidation: action resolution, the content gate, dispatch."""

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
        status, resolved, spy = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "noop_no_stored_keys"
        assert resolved is ConsolidationAction.RECONCILE
        assert spy.call_count == 0
        assert fresh_store.tiers_with_registry() == [], (
            "the gate must not mint a registry while answering the noop question"
        )

    def test_reconcile_dispatches_with_no_store_resident(self, tmp_path, monkeypatch) -> None:
        """No live ``memory_store`` (pre-boot/test posture) is unprovable — it dispatches."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        assert "memory_store" not in state, "fixture sanity: no store seeded"

        status, resolved, spy = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE

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
        status, resolved, spy = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE

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
        status, resolved, spy = _dispatch(
            state, ConsolidationAction.RECONCILE, monkeypatch=monkeypatch
        )

        assert status == "started_full"
        assert resolved is ConsolidationAction.RECONCILE

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
        status, resolved, spy = _dispatch(
            state, ConsolidationAction.INTERIM, monkeypatch=monkeypatch
        )

        assert status == "started"
        assert resolved is ConsolidationAction.INTERIM
        assert spy.submitted == [app_module._extract_and_start_training]


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
        a scheduled tick would find no full fold due, but ``/consolidate``
        requests ``FULL`` directly and never consults the deadline —
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


class TestReconsolidatePendingRecordResume:
    """``/reconsolidate`` has no special relationship to a pending
    consolidation event's record: it never discards one, and it is not a
    recovery or abandon door.  A pending record is resumed and finished
    first, exactly like the other three consolidation endpoints -- the
    identical resume-pending-first contract, never a RECONCILE-only
    carve-out."""

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
        all, so no full fold is due) and noops for lack of pending
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

        An empty rebuild is a noop no matter who dispatched it -- there is
        no on-demand-fold exemption from this guard -- and the
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
        """Control: a completed (non-aborted) result records outcome
        ``full_trained``, stamps ``last_consolidation``, and resolves
        incidents."""
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
        """Control: a completed interim result resolves the training_crash incident."""
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
# TestFiveDoorPendingRecordGuard — the pending-record verdict
# (``deferred_event_pending`` -> ``consolidation_pending``) answered
# identically by every mutating door that reads ``active_consolidation()``.
# Each door's own five BUSY arms (consolidating / bg-training / cloud-only /
# trial-active / base-swap-active) live with that door's own test file; this
# class pins the SIXTH, pending-record arm across all five doors in one
# place.
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
    status, resolved = app_module._dispatch_consolidation(
        action, reason=_reason_for(action), spec=spec
    )
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
        """Control: retro-claim runs for staging actions too."""
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


# ---------------------------------------------------------------------------
# TestUnreadableLedgerRefusal — a stage ledger present on disk but this
# process cannot interpret it is a refusal, never "nothing pending": the
# HTTP door surface only (the arbitrator itself is exercised through
# _route_client's real dispatch, never called directly).
# ---------------------------------------------------------------------------


class TestUnreadableLedgerRefusal:
    def test_consolidate_answers_deferred_event_unreadable_and_records_an_incident(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.incidents import read_incidents

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "stage_ledger.json").write_bytes(b"not json at all {{{")
        extraction_dir = state_dir / "extraction"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        marker = extraction_dir / "marker.txt"
        marker.write_text("untouched")

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json()["status"] == "deferred_event_unreadable"
        assert submitted == []

        incidents = read_incidents(state_dir)
        matching = [i for i in incidents if i.type == "stage_ledger_unreadable"]
        assert len(matching) == 1
        assert matching[0].id == "stage_ledger_unreadable:stage_ledger"

        assert marker.exists()
        assert marker.read_text() == "untouched"

    def test_interim_discard_refuses_409_over_an_unreadable_pending_record(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from tests.server._state_builders import _make_discard_state

        data_dir = tmp_path / "data"
        state_dir = data_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "stage_ledger.json").write_bytes(b"not json at all {{{")
        state = _make_discard_state(tmp_path)

        monkeypatch.setattr(app_module, "_state", state)

        from fastapi.testclient import TestClient

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/interim/discard", json={"confirm": True})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidation_pending"


# ---------------------------------------------------------------------------
# TestScheduledTickFullWindow / TestScheduledTickOutsideWindow /
# TestManualDoorsAndCountZeroIgnoreWindow — full_window gates a TIMER
# firing's own resolution to FULL, never a manual door's, and never the
# count-zero cadence-is-the-schedule path.
# ---------------------------------------------------------------------------


@pytest.fixture()
def berlin_timezone(monkeypatch):
    """Pin the process timezone to Europe/Berlin for a window/cadence test,
    restored (env and C library state alike) afterward."""
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _seed_stale_cadence_mark(tmp_path, *, before: float) -> None:
    """Write a cadence mark well before *before* so the cadence reads DUE,
    without seeding a full-fold-start mark."""
    from paramem.server.schedule_state import ScheduleMarks, write_marks

    write_marks(
        tmp_path / "state",
        ScheduleMarks(last_cadence_mark_epoch=before - 100_000, last_full_start_epoch=None),
    )


class TestScheduledTickFullWindow:
    """A full fold due inside ``full_window`` starts once per opening."""

    def test_full_fold_starts_inside_the_window_and_stamps_the_start_mark(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_marks

        fixed_now = datetime(2026, 1, 6, 1, 30).timestamp()
        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, refresh_cadence="12h", period_seconds=3600
        )
        _make_interim_slot(state["config"].adapter_dir, "20260105T2330", payload="weights")
        _seed_stale_cadence_mark(tmp_path, before=fixed_now)
        monkeypatch.setattr(app_module.time, "time", lambda: fixed_now)

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "started_full"
        assert _submitted_full_fold_events(spy) == ["full"]
        marks = read_marks(tmp_path / "state")
        assert marks.last_full_start_epoch == fixed_now

    def test_a_second_tick_in_the_same_opening_noops_and_submits_nothing(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        first_now = datetime(2026, 1, 6, 1, 30).timestamp()
        second_now = datetime(2026, 1, 6, 2, 30).timestamp()
        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, refresh_cadence="12h", period_seconds=3600
        )
        _make_interim_slot(state["config"].adapter_dir, "20260105T2330", payload="weights")
        _seed_stale_cadence_mark(tmp_path, before=first_now)
        monkeypatch.setattr(app_module.time, "time", lambda: first_now)
        _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        # The first dispatch's own executor submission set consolidating=True
        # in production; nothing runs the (spied-out) fold to clear it again.
        state["consolidating"] = False
        monkeypatch.setattr(app_module.time, "time", lambda: second_now)
        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "noop_not_due"
        assert spy.call_count == 0


class TestScheduledTickOutsideWindow:
    """The same due full fold, but ``full_window`` is shut."""

    def test_full_fold_due_but_window_shut_answers_noop_outside_window(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module
        from paramem.memory.interim_adapter import iter_interim_dirs
        from paramem.server.app import ConsolidationAction

        fixed_now = datetime(2026, 1, 6, 12, 0).timestamp()
        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, refresh_cadence="12h", period_seconds=3600
        )
        _make_interim_slot(state["config"].adapter_dir, "20260105T2330", payload="weights")
        _seed_stale_cadence_mark(tmp_path, before=fixed_now)
        monkeypatch.setattr(app_module.time, "time", lambda: fixed_now)

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "noop_outside_window"
        assert spy.call_count == 0
        assert len(list(iter_interim_dirs(state["config"].adapter_dir, payload_only=True))) == 1, (
            "a firing the window shuts must mint no new interim slot"
        )


class TestManualDoorsAndCountZeroIgnoreWindow:
    """A manual door, and the count-zero cadence-is-the-schedule path, never
    read ``full_window``."""

    def test_consolidate_dispatches_full_outside_the_window(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module

        fixed_now = datetime(2026, 1, 6, 12, 0).timestamp()
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="12h")
        _make_interim_slot(state["config"].adapter_dir, "20260701T0000", payload="weights")
        monkeypatch.setattr(app_module.time, "time", lambda: fixed_now)

        client, submitted = _route_client(state, monkeypatch)
        resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json() == {"status": "started_full", "action": "full"}
        assert _route_events(submitted) == ["full"]

    def test_scheduled_tick_dispatches_full_and_consumes_the_mark_at_count_zero(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction
        from paramem.server.schedule_state import read_marks

        fixed_now = datetime(2026, 1, 6, 12, 0).timestamp()
        state = _make_arbitrator_state(
            tmp_path, max_interim_count=0, refresh_cadence="12h", named_sessions=1
        )
        _seed_stale_cadence_mark(tmp_path, before=fixed_now)
        stale_mark = read_marks(tmp_path / "state").last_cadence_mark_epoch
        monkeypatch.setattr(app_module.time, "time", lambda: fixed_now)

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "started_full"
        assert _submitted_full_fold_events(spy) == ["full"]
        marks = read_marks(tmp_path / "state")
        assert marks.last_cadence_mark_epoch is not None
        assert marks.last_cadence_mark_epoch > stale_mark, (
            "a FULL dispatch standing on a due cadence mark at count zero must advance it"
        )


# ---------------------------------------------------------------------------
# TestArbitratorPreStagesAndMigrationPreempt — the pre-stages
# (retro-claim, session triage/retirement) and the pending-migration
# pre-empt run on every TIMER firing whatever the decider's verdict, but a
# BOOT or IDLE firing whose verdict is a noop walks none of them, and a
# pending ledger's own resume/defer step ends the resolution before the
# migration pre-empt is ever reached.
# ---------------------------------------------------------------------------


class TestArbitratorPreStagesAndMigrationPreempt:
    def test_not_due_timer_firing_with_no_ledger_starts_the_pending_migration(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="")
        state["pending_rehydration"] = True

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "started_migration"
        assert spy.submitted == [app_module._run_active_store_migration_sync]

    def test_ledger_pending_with_resume_deferred_starts_no_migration(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="")
        state["pending_rehydration"] = True
        # A conversation used the model 100s ago: past the idle debounce
        # (training_idle_debounce_s=30 in this fixture) so the debounce
        # itself does not answer first, but well short of the idle timeout
        # (10 minutes), so the interim event's resume rule cannot be met yet.
        state["last_model_use_monotonic"] = time.monotonic() - 100

        spy = _ExecutorSpy()
        state["event_loop"] = spy.loop
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)

        async def _run() -> "tuple[str, ConsolidationAction]":
            result = app_module._dispatch_consolidation(
                ConsolidationAction.AUTO, reason=_reason_for(ConsolidationAction.AUTO)
            )
            # The deferral this dispatch earns is owned by IDLE, which arms
            # the idle watch task -- drain it inside this loop so the test
            # leaves no task pending when asyncio.run() closes the loop.
            task = state.get("idle_watch_task")
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            return result

        status, _resolved = asyncio.run(_run())

        assert status == "deferred_resume_waiting"
        assert spy.call_count == 0

    def test_resumable_ledger_with_idle_server_resumes_and_starts_no_migration(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="")
        state["pending_rehydration"] = True

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "started_resume"
        assert spy.submitted == [app_module._run_pending_event_resume]

    def test_not_due_boot_firing_with_no_ledger_starts_nothing_and_leaves_sessions_pending(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.app import ConsolidationAction
        from paramem.server.consolidation_choice import DispatchReason

        state = _make_arbitrator_state(
            tmp_path, max_interim_count=7, refresh_cadence="", anon_sessions=2
        )
        state["pending_rehydration"] = True
        assert len(state["session_buffer"].pending_facts()) == 2, "fixture sanity"

        calls: list[int] = []
        real_retro_claim = app_module._retro_claim_orphan_sessions

        def _spy() -> int:
            calls.append(1)
            return real_retro_claim()

        spy = _ExecutorSpy()
        state["event_loop"] = spy.loop
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", _spy)

        status, _resolved = app_module._dispatch_consolidation(
            ConsolidationAction.AUTO, reason=DispatchReason.BOOT
        )

        assert status == "noop_not_due"
        assert spy.call_count == 0, "a BOOT noop must start no migration and dispatch nothing"
        assert calls == [], "the pre-stages (including retro-claim) must not run on a BOOT noop"
        assert len(state["session_buffer"].pending_facts()) == 2, (
            "a BOOT firing whose verdict is a noop must retire no session"
        )

    def test_calibrate_dispatch_with_ledger_pending_runs_as_calibration(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.consolidation_action import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="")
        state["pending_rehydration"] = True

        status, _resolved, spy, calls = _dispatch_with_retro_claim_spy(
            state, ConsolidationAction.CALIBRATE, monkeypatch=monkeypatch, spec=MagicMock()
        )

        assert status == "started_calibration"
        assert calls == [1]
        assert spy.call_count == 1
        assert app_module._run_active_store_migration_sync not in spy.submitted, (
            "the migration pre-empt must be skipped while an event is pending, "
            "so the calibrate run the operator asked for is what dispatches"
        )

    def test_not_due_timer_firing_with_no_ledger_and_a_failed_integrity_check_refuses(
        self, tmp_path, monkeypatch
    ) -> None:
        """The same not-due TIMER firing with a pending migration that starts
        it (``test_not_due_timer_firing_with_no_ledger_starts_the_pending_migration``
        above) instead answers ``migration_skipped_degraded`` and starts
        nothing when the boot-time integrity check failed."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, refresh_cadence="")
        state["pending_rehydration"] = True
        state["integrity_check_failed"] = True

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "migration_skipped_degraded"
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# TestIdleDebounce — the idle debounce (config.consolidation.
# training_idle_debounce_s, _state["last_model_use_monotonic"]) is read for
# every action, ahead of the schedule's own gates — a safety property, not a
# schedule, so an explicit OPERATOR request defers on it too.
# ---------------------------------------------------------------------------


class TestIdleDebounce:
    def test_a_recent_model_use_defers_a_timer_firing(self, tmp_path, monkeypatch) -> None:
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["last_model_use_monotonic"] = time.monotonic() - 5  # < 30s debounce

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "deferred_model_in_use"
        assert spy.call_count == 0

    def test_a_recent_model_use_defers_an_operator_door_too(self, tmp_path, monkeypatch) -> None:
        """The debounce is a safety property, not a schedule condition —
        an explicit OPERATOR request (``/consolidate``) defers on it exactly
        like a scheduled TIMER firing."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["last_model_use_monotonic"] = time.monotonic() - 5

        status, _resolved, spy = _dispatch(state, ConsolidationAction.FULL, monkeypatch=monkeypatch)

        assert status == "deferred_model_in_use"
        assert spy.call_count == 0

    def test_a_model_use_older_than_the_debounce_window_proceeds(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["last_model_use_monotonic"] = time.monotonic() - 60  # > 30s debounce

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status != "deferred_model_in_use"
        # Nothing pending -> the schedule resolves AUTO into INTERIM (a due
        # cadence tick with a ring), whose content gate then noops -- proof
        # the dispatch actually reached the schedule's own gates rather than
        # stopping at the debounce.
        assert status == "noop_no_pending"

    def test_a_zero_debounce_disables_the_gate_even_moments_after_model_use(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["config"].consolidation.training_idle_debounce_s = 0
        state["last_model_use_monotonic"] = time.monotonic()  # just now

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status != "deferred_model_in_use"
        assert status == "noop_no_pending"

    def test_a_clock_never_stamped_proceeds_regardless_of_the_debounce_window(
        self, tmp_path, monkeypatch
    ) -> None:
        """``last_model_use_monotonic is None`` (a server untouched since
        boot) is read as idle, never as "just used" -- the debounce never
        fires on it."""
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        assert state["last_model_use_monotonic"] is None, "fixture sanity"

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status != "deferred_model_in_use"
        assert status == "noop_no_pending"


# ---------------------------------------------------------------------------
# TestFullConsolidationOverdueIncident — _record_full_consolidation_overdue:
# the oldest un-folded interim slot's age against 2x the full-fold period
# decides whether the "full_consolidation_overdue" incident fires. Shared by
# a direct FULL dispatch and a resumed FULL ledger (_dispatch_resume).
# ---------------------------------------------------------------------------


class TestFullConsolidationOverdueIncident:
    def test_a_full_dispatch_past_its_runway_records_the_overdue_incident(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.app import ConsolidationAction
        from paramem.server.incidents import read_incidents

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, period_seconds=3600)
        _make_interim_slot(state["config"].adapter_dir, "20200101T0000", payload="weights")

        status, _resolved, spy = _dispatch(state, ConsolidationAction.FULL, monkeypatch=monkeypatch)

        assert status == "started_full"
        active = [inc for inc in read_incidents(tmp_path / "state") if inc.status == "active"]
        assert any(inc.type == "full_consolidation_overdue" for inc in active)

    def test_a_full_dispatch_within_its_runway_records_no_overdue_incident(
        self, tmp_path, monkeypatch
    ) -> None:
        from datetime import datetime

        from paramem.server.app import ConsolidationAction
        from paramem.server.incidents import read_incidents

        state = _make_arbitrator_state(tmp_path, max_interim_count=7, period_seconds=1_000_000)
        fresh_stamp = datetime.now().strftime("%Y%m%dT%H%M")
        _make_interim_slot(state["config"].adapter_dir, fresh_stamp, payload="weights")

        status, _resolved, spy = _dispatch(state, ConsolidationAction.FULL, monkeypatch=monkeypatch)

        assert status == "started_full"
        active = [inc for inc in read_incidents(tmp_path / "state") if inc.status == "active"]
        assert not any(inc.type == "full_consolidation_overdue" for inc in active)

    def test_a_resumed_full_ledger_past_its_runway_records_the_same_incident_type(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.app import ConsolidationAction
        from paramem.server.incidents import read_incidents

        _write_pending_ledger(tmp_path, event="full")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7, period_seconds=3600)
        _make_interim_slot(state["config"].adapter_dir, "20200101T0000", payload="weights")

        status, _resolved, spy = _dispatch(state, ConsolidationAction.FULL, monkeypatch=monkeypatch)

        assert status == "started_resume"
        active = [inc for inc in read_incidents(tmp_path / "state") if inc.status == "active"]
        assert any(inc.type == "full_consolidation_overdue" for inc in active)
