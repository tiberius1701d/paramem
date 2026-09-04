"""Store-quarantine cross-cutting pins.

A broken memory store quarantines the STORE, not the process: boot
completes with ``_state["memory_store"]`` unset, every other subsystem
keeps serving, and only the doors that read or write the live
``MemoryStore`` refuse — with the ``"store_quarantined"`` verdict, mapped
through the same ``refusal_for`` vocabulary every other consolidation
refusal uses. This file exercises that propagation across the otherwise
unrelated endpoint modules it touches (the consolidation arbitrator,
``POST /speaker/forget``, ``POST /interim/discard``, the migration/trial
doors, ``GET /status``, ``POST /debug/erase-keys``, ``POST /backup/restore``,
the base-swap branch of ``POST /migration/rollback``) — a single
cross-cutting concern, not a natural fit for any one of those endpoints'
own test files.

Three doors stay open on an EXISTING quarantine by design, for different
reasons: ``POST /migration/rollback`` never touches the live store at all
(non-base-swap branch) or is itself the recovery flow (base-swap branch);
``POST /debug/erase-keys`` is a file surgeon that can repair the very
condition that quarantined the store; ``POST /backup/restore`` is the other
recovery door and both attempts a deliberate re-entry (updating the cause)
and the LIFT that exits quarantine.

Convention: TestClient without lifespan; ``_state`` monkeypatched per test
— mirrors ``tests/server/test_attention_status_e2e.py`` and
``tests/server/test_speaker_forget.py``. The consolidation-arbitrator pins
call ``_dispatch_consolidation`` directly, mirroring
``tests/server/test_consolidate_dispatch.py``'s ``_make_arbitrator_state``
pattern.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from tests._fold_fixtures import (
    _make_loop,
    _recalled_entries_from_store,
    _rel,
    _wire_fakes,
)
from tests._fold_fixtures import _make_state as _make_resume_state
from tests.server._state_builders import _write_pending_ledger
from tests.server.test_consolidate_dispatch import _dispatch, _make_arbitrator_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CAUSE = {
    "cause": {
        "exception_type": "TierBindingUnpublishable",
        "message": (
            "tier registry binding unverified for publish: "
            "'episodic' (no_matching_slot: stale hash)"
        ),
    },
    "quarantined_at": "2026-08-16T00:00:00Z",
}


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _make_door_state(tmp_path: Path, *, quarantined: bool) -> dict:
    """Minimal ``_state`` for a store-mutating door: the quarantine check
    (composed at the door's own call site, or first in the arbitrator) is
    the very first thing the handler reads, so nothing else needs seeding."""
    return {
        "config": _make_config(tmp_path),
        "consolidation_loop": None,
        "speaker_store": MagicMock(),
        "session_buffer": MagicMock(),
        "mode": "local",
        "consolidating": False,
        "background_trainer": None,
        "migration": None,
        "router": MagicMock(),
        "adapter_manifest_status": {},
        "model": MagicMock(),
        "tokenizer": None,
        "memory_store": None,
        "store_quarantine": dict(_CAUSE) if quarantined else None,
    }


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# The consolidation arbitrator — all four actions
# ---------------------------------------------------------------------------


class TestArbitratorRejection:
    """``_dispatch_consolidation`` refuses every action while the store is
    quarantined AND nothing is pending, before any of its other guards run.

    A PENDING event resumes ahead of this verdict instead — the resume
    needs nothing from the live store, and completing it is the mechanism
    that heals a quarantine caused by a crashed publish on a cold-born
    tier. See ``TestQuarantinedPendingResumeHeal`` below for that arm;
    this class covers only the case the verdict itself still answers:
    quarantined with nothing pending to resume."""

    @pytest.mark.parametrize("action_name", ["AUTO", "FULL", "INTERIM", "RECONCILE"])
    def test_quarantined_store_defers_every_action(
        self, tmp_path, monkeypatch, action_name
    ) -> None:
        from paramem.server.app import ConsolidationAction

        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["store_quarantine"] = dict(_CAUSE)

        status, resolved, spy = _dispatch(
            state, getattr(ConsolidationAction, action_name), monkeypatch=monkeypatch
        )

        assert status == "deferred_store_quarantined"
        assert resolved is getattr(ConsolidationAction, action_name)
        assert spy.submitted == []

    def test_healthy_store_does_not_trip_the_quarantine_verdict(self, monkeypatch) -> None:
        """A healthy store (``store_quarantine`` absent or ``None``) never
        returns the quarantine verdict — proven via the shared predicate
        directly, so this pin does not depend on which guard fires next."""
        monkeypatch.setattr(app_module, "_state", {"store_quarantine": None})
        assert app_module._store_quarantine_verdict() is None

        monkeypatch.setattr(app_module, "_state", {})
        assert app_module._store_quarantine_verdict() is None


# ---------------------------------------------------------------------------
# The arbitrator's resume-pending-first arm dispatches a readable pending
# ledger AHEAD OF the store-quarantine verdict (``_dispatch_consolidation``'s
# docstring, steps 3-5) — so a quarantined store with something pending
# still resumes rather than refusing. The busy guards and the idle debounce
# (steps 1-2) still run ahead of the resume itself, and a withheld main
# tier's binding-unverified gate (step 7, below the resume) never reaches a
# resumed dispatch either.
# ---------------------------------------------------------------------------


class TestQuarantinedPendingResumeDispatchesAheadOfTheGate:
    def test_a_quarantined_store_with_a_readable_pending_ledger_resumes_while_idle(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["store_quarantine"] = dict(_CAUSE)
        assert state["last_model_use_monotonic"] is None, "fixture sanity: idle by default"

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "started_resume"
        assert spy.submitted == [app_module._run_pending_event_resume]

    def test_a_withheld_main_tier_does_not_block_a_quarantined_pending_resume(
        self, tmp_path, monkeypatch
    ) -> None:
        """The tier-unverified gate (``adapter_manifest_status``) lives
        BELOW the resume-pending-first arm, so a main tier flagged
        unverified never reaches a dispatch that resumes instead."""
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["store_quarantine"] = dict(_CAUSE)
        state["adapter_manifest_status"] = {
            "episodic": {"status": "no_matching_slot", "severity": "red"}
        }

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "started_resume"
        assert spy.submitted == [app_module._run_pending_event_resume]

    @pytest.mark.parametrize(
        "field, value, expected",
        [
            ("migration", {"base_swap_active": True}, "deferred_base_swap_active"),
            ("consolidating", True, "deferred_already_running"),
            ("mode", "cloud-only", "deferred_cloud_only"),
            ("background_trainer", None, "deferred_bg_training"),  # replaced below
            ("migration", {"state": "TRIAL"}, "deferred_trial_active"),
        ],
        ids=[
            "base_swap_active",
            "already_running",
            "cloud_only",
            "bg_training",
            "trial_active",
        ],
    )
    def test_each_busy_guard_answers_before_a_quarantined_pending_resume(
        self, tmp_path, monkeypatch, field, value, expected
    ) -> None:
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["store_quarantine"] = dict(_CAUSE)
        if field == "background_trainer":
            state["background_trainer"] = MagicMock(is_training=True)
        else:
            state[field] = value

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == expected
        assert spy.submitted == []

    def test_the_idle_debounce_answers_before_a_quarantined_pending_resume(
        self, tmp_path, monkeypatch
    ) -> None:
        from paramem.server.app import ConsolidationAction

        _write_pending_ledger(tmp_path, event="interim")
        state = _make_arbitrator_state(tmp_path, max_interim_count=7)
        state["store_quarantine"] = dict(_CAUSE)
        state["last_model_use_monotonic"] = time.monotonic() - 5  # < 30s debounce

        status, _resolved, spy = _dispatch(state, ConsolidationAction.AUTO, monkeypatch=monkeypatch)

        assert status == "deferred_model_in_use"
        assert spy.submitted == []


# ---------------------------------------------------------------------------
# Resume-pending-first heals a quarantine: a pending event resumes ahead of
# the store-quarantine verdict and the tier-unverified gate
# (``_dispatch_consolidation``'s docstring, steps 3-5), and its own
# ``all_live`` completion lifts the quarantine in-band, no restart or
# restore needed.
# ---------------------------------------------------------------------------


def _stage_pending_interim_event(tmp_path: Path, *, venue: str):
    """Build a real ``ConsolidationLoop`` and stage one interim event,
    stopping immediately after staging (nothing written, nothing published) —
    the earliest possible pending shape: a crash right after the extraction
    stage was recorded. Returns ``(loop, staged)``."""
    loop = _make_loop(tmp_path, resident_tiers=["episodic"])
    staged = loop.stage_event(
        recalled_entries=_recalled_entries_from_store(loop),
        event="interim",
        venue=venue,
        stamp="stamp1",
        primary_tiers={"episodic": "episodic"},
        candidate_tiers={},
        episodic_rels=[_rel("alex", "likes", "coffee")],
        session_ids=["s1"],
    )
    assert staged is not None
    assert "episodic" in staged.built_tiers
    return loop, staged


def _fake_lift_publishes(monkeypatch, new_store) -> list:
    """Stub ``_lift_quarantined_store`` to mirror its real success contract:
    publish ``_state["memory_store"]`` and clear the quarantine marker (the
    clear normally happens inside ``_hydrate_memory_store_in_place``, which
    this test never drives for real).

    Returns the list of ``config`` objects the stub was invoked with, so a
    caller can assert the lift was (or was not) attempted.
    """
    calls: list = []

    def _fake_lift(config):
        calls.append(config)
        app_module._state["memory_store"] = new_store
        app_module._state["store_quarantine"] = None
        return True

    monkeypatch.setattr(app_module, "_lift_quarantined_store", _fake_lift)
    return calls


class TestQuarantineLiftGpuLock:
    """The lift's source medium (``_build_store_contents``)
    follows ``config.consolidation.mode``, not ``staged_event.venue`` — a
    disk-venue ledger can still resolve to a GPU-touching
    ``WeightMemorySource`` fill. The disk venue's lift call must take
    ``gpu_lock_sync()`` itself (nothing holds the lock on entry); the
    weights venue's lift call must stay bare (its caller already holds the
    non-reentrant ``_gpu_thread_lock`` for the whole cycle, and
    re-acquiring would deadlock it)."""

    @pytest.mark.parametrize(
        "venue,expect_lock", [("disk", True), ("weights", False)], ids=["disk", "weights"]
    )
    def test_lift_lock_is_venue_conditional(
        self, tmp_path, monkeypatch, venue, expect_lock
    ) -> None:
        from paramem.memory.store import MemoryStore

        loop, staged = _stage_pending_interim_event(tmp_path, venue=venue)
        _wire_fakes(loop, monkeypatch)

        state = _make_resume_state(loop, tmp_path=tmp_path)
        state["memory_store"] = None
        state["store_quarantine"] = dict(_CAUSE)
        monkeypatch.setattr(app_module, "_state", state)

        new_store = MemoryStore()
        _fake_lift_publishes(monkeypatch, new_store)

        lock_calls: list = []

        class _SpyLockCtx:
            def __enter__(self):
                lock_calls.append("enter")
                return self

            def __exit__(self, *exc):
                lock_calls.append("exit")
                return False

        import paramem.server.gpu_lock as gpu_lock_mod

        monkeypatch.setattr(gpu_lock_mod, "gpu_lock_sync", lambda *a, **kw: _SpyLockCtx())

        result = app_module._finish_resumed_event(loop, staged, router=state["router"])

        assert result["completed"] is True
        assert bool(lock_calls) is expect_lock, (
            f"venue={venue}: expected gpu_lock_sync() to be "
            f"{'acquired' if expect_lock else 'left untouched'} around the lift"
        )
        if expect_lock:
            assert lock_calls == ["enter", "exit"]
        assert app_module._state["memory_store"] is new_store


# ---------------------------------------------------------------------------
# POST /speaker/forget, POST /interim/discard — rejected
# ---------------------------------------------------------------------------


class TestSpeakerForgetAndInterimDiscardRejection:
    def test_speaker_forget_refuses_409(self, tmp_path, monkeypatch) -> None:
        state = _make_door_state(tmp_path, quarantined=True)
        client = _make_client(monkeypatch, state)

        resp = client.post("/speaker/forget", json={"speaker_id": "speaker0"})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "store_quarantined"

    def test_interim_discard_refuses_409(self, tmp_path, monkeypatch) -> None:
        state = _make_door_state(tmp_path, quarantined=True)
        client = _make_client(monkeypatch, state)

        resp = client.post("/interim/discard", json={"tier": "episodic_interim_20260101T0000"})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "store_quarantined"

    def test_speaker_forget_unaffected_when_healthy(self, tmp_path, monkeypatch) -> None:
        """A healthy store never surfaces the quarantine verdict — the door
        falls through to its normal guard chain instead (here: no loop
        configured, so speaker_store.remove is never reached and the
        request proceeds past the 409 this test pins against)."""
        state = _make_door_state(tmp_path, quarantined=False)
        client = _make_client(monkeypatch, state)

        resp = client.post("/speaker/forget", json={"speaker_id": "speaker0"})

        assert resp.status_code != 409 or resp.json()["detail"]["error"] != "store_quarantined"


# ---------------------------------------------------------------------------
# Migration / trial doors
# ---------------------------------------------------------------------------


class TestMigrationDoorRejection:
    """``POST /migration/confirm`` and ``POST /migration/accept`` read or
    trigger a re-hydration of the live memory store, so both refuse while
    it is quarantined — checked before any of their own state-machine
    preconditions. ``POST /migration/rollback`` does NOT touch the live
    store (config/adapter-archive rotation only) and is the recovery exit
    from a bad trial, so it deliberately stays open — pinned below."""

    def test_migration_confirm_refuses_409(self, tmp_path, monkeypatch) -> None:
        state = _make_door_state(tmp_path, quarantined=True)
        client = _make_client(monkeypatch, state)

        resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "store_quarantined"

    def test_migration_accept_refuses_409(self, tmp_path, monkeypatch) -> None:
        state = _make_door_state(tmp_path, quarantined=True)
        client = _make_client(monkeypatch, state)

        resp = client.post("/migration/accept")

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "store_quarantined"

    def test_migration_rollback_ignores_quarantine(self, tmp_path, monkeypatch) -> None:
        """Rollback never touches the live store, so a quarantined store
        does not block it — the request reaches rollback's OWN precondition
        check instead (``not_found``, no trial active in this fixture)."""
        from paramem.server.migration import initial_migration_state

        state = _make_door_state(tmp_path, quarantined=True)
        state["migration"] = initial_migration_state()
        client = _make_client(monkeypatch, state)

        resp = client.post("/migration/rollback")

        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "not_found"


# ---------------------------------------------------------------------------
# Admin doors keep serving
# ---------------------------------------------------------------------------


class TestAdminDoorsServeThroughQuarantine:
    def test_status_reports_the_quarantine_and_still_serves(self, tmp_path, monkeypatch) -> None:
        from paramem.server.migration import initial_migration_state

        cfg = _make_config(tmp_path)
        cfg.model_name = "mistral"
        cfg.model_config.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        cfg.adapters.episodic.enabled = False
        cfg.adapters.semantic.enabled = False
        cfg.adapters.procedural.enabled = False
        cfg.consolidation.refresh_cadence = ""
        cfg.consolidation.interim_resume = "immediate"
        cfg.consolidation.full_window = "01:00-04:00"
        cfg.consolidation.consolidation_period_string = ""
        cfg.consolidation.max_interim_count = 0
        cfg.consolidation.mode = "train"
        cfg.consolidation.quiet_hours_mode = "always_off"
        cfg.consolidation.quiet_hours_start = "00:00"
        cfg.consolidation.quiet_hours_end = "00:00"
        cfg.consolidation.training_temp_limit = 0
        cfg.paths.data.mkdir(parents=True, exist_ok=True)
        cfg.security.backups.max_total_disk_gb = 20.0

        buf = MagicMock()
        buf.get_summary.return_value = {
            "total": 0,
            "orphaned": 0,
            "oldest_age_seconds": None,
            "per_speaker": {},
            "per_source_type": {},
        }

        state = {
            "model": None,
            "tokenizer": None,
            "config": cfg,
            "config_path": str(tmp_path / "server.yaml"),
            "session_buffer": buf,
            "speaker_store": None,
            "router": None,
            "cloud_agent": None,
            "ha_client": None,
            "consolidation_loop": None,
            "consolidating": False,
            "last_consolidation": None,
            "background_trainer": None,
            "mode": "local",
            "cloud_only_reason": None,
            "tts_manager": None,
            "stt": None,
            "speaker_embedding_backend": None,
            "unknown_speakers": {},
            "pending_enrollments": set(),
            "migration": initial_migration_state(),
            "server_started_at": "2026-08-16T00:00:00+00:00",
            "config_drift": {},
            "adapter_manifest_status": {},
            "last_consolidation_result": None,
            "memory_store": None,
            "store_quarantine": dict(_CAUSE),
        }
        client = _make_client(monkeypatch, state)

        resp = client.get("/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["store_quarantined"]["cause"]["exception_type"] == "TierBindingUnpublishable"

    def test_debug_erase_keys_ignores_quarantine(self, tmp_path, monkeypatch) -> None:
        """A quarantined store does not close ``POST /debug/erase-keys`` —
        the request reaches the door's OWN confirmation gate instead of a
        ``store_quarantined`` 409. The door is a file surgeon that can
        repair the very condition that quarantined the store (see
        ``tests/server/test_debug_erase_keys_endpoint.py`` for the file
        surgery + lift coverage this file's cross-cutting scope does not
        duplicate)."""
        state = _make_door_state(tmp_path, quarantined=True)
        state["config"].debug = True
        client = _make_client(monkeypatch, state)

        resp = client.post("/debug/erase-keys", json={"keys": ["some_key"]})

        assert resp.status_code != 409 or resp.json()["detail"]["error"] != "store_quarantined"
        assert resp.json()["detail"]["error"] == "confirmation_required"


# ---------------------------------------------------------------------------
# POST /backup/restore — the other recovery door, stays open on quarantine
# ---------------------------------------------------------------------------


class TestBackupRestoreIgnoresQuarantine:
    def test_backup_restore_ignores_existing_quarantine(self, tmp_path, monkeypatch) -> None:
        """A quarantined store does not close ``POST /backup/restore`` — the
        request reaches the door's own ``not_found`` precondition instead of
        a ``store_quarantined`` 409 (no backup slot exists in this fixture,
        so it never reaches the point of re-entering quarantine)."""
        from paramem.server.migration import initial_migration_state

        state = _make_door_state(tmp_path, quarantined=True)
        state["migration"] = initial_migration_state()
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": "does-not-exist"})

        assert resp.status_code != 409 or resp.json()["detail"]["error"] != "store_quarantined"
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "not_found"
