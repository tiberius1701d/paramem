"""Route-level tests for the ten ``/calibrate/*`` doors and the shared
dispatch envelope they run under.

Drives the REAL route functions (``TestClient(app_module.app)``) and the
REAL arbitrator (``_dispatch_consolidation`` runs unmodified); only the
executor submission (``_dispatch_to_executor``) is stubbed so nothing
actually reaches the GPU — the same seam ``tests/server/test_consolidate_
dispatch.py``'s ``_route_client`` uses for the four consolidation routes.
No model load, no real extraction: these tests pin the BOUNDARY contract
(status, identity, guard deferrals, submission count), not extraction
correctness (already covered by ``tests/server/test_calibrate.py``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tests.server.test_consolidate_dispatch import _make_arbitrator_state

# ---------------------------------------------------------------------------
# Per-route valid minimal payloads.
# ---------------------------------------------------------------------------


def _turn_marked_transcript() -> str:
    """A transcript starting with the REAL production user-turn marker.

    Derived from the same renderer ``_require_turn_marked_transcript``
    checks against, rather than a hardcoded ``"[user]"`` literal that could
    silently drift from the production surface.
    """
    from paramem.server.calibrate import _production_turn_markers

    marker = _production_turn_markers()[0]
    return f"{marker} hello"


def _session_graph_dict() -> dict:
    from paramem.graph.schema import SessionGraph

    return SessionGraph(session_id="calib", timestamp="2026-01-01T00:00:00Z").model_dump(
        mode="json"
    )


def _payload_for(route: str) -> dict:
    transcript = _turn_marked_transcript()
    if route in ("extract", "procedural"):
        return {"transcript": transcript, "speaker_id": "speaker1"}
    if route in ("anonymize", "enrich", "plausibility"):
        return {
            "transcript": transcript,
            "speaker_id": "speaker1",
            "graph": _session_graph_dict(),
        }
    if route == "normalize":
        return {"relations": [{"subject": "alex", "predicate": "lives in", "object": "berlin"}]}
    if route == "anonymize_facts":
        return {"facts": [{"subject": "alex", "predicate": "lives in", "object": "berlin"}]}
    if route == "name":
        return {"turns": [{"role": "user", "text": "hi, I'm Alex"}]}
    if route == "respond":
        return {"text": "hello", "speaker_id": "speaker1"}
    if route == "extract_pending":
        return {}
    raise AssertionError(f"no payload declared for route {route!r}")


_ALL_CALIBRATE_ROUTES = (
    "extract",
    "procedural",
    "anonymize",
    "enrich",
    "plausibility",
    "normalize",
    "anonymize_facts",
    "name",
    "respond",
    "extract_pending",
)


# ---------------------------------------------------------------------------
# State / client builders
# ---------------------------------------------------------------------------


def _make_calibrate_state(
    tmp_path, *, calibrate_enabled: bool = True, named_sessions: int = 0, **overrides
) -> dict:
    """``_state`` wired for a live calibrate dispatch: real arbitrator state
    (:func:`_make_arbitrator_state`) plus the calibrate-specific handles
    every route's ``preflight``/``validate_*`` reads.

    ``named_sessions`` seeds pending NAMED sessions in the real
    ``SessionBuffer`` — required for ``/calibrate/extract_pending`` to
    answer ``started_calibration`` rather than ``noop_no_pending`` (its
    content is the identical pending-NAMED-set check ``/consolidate/interim``
    applies).
    """
    state = _make_arbitrator_state(tmp_path, max_interim_count=7, named_sessions=named_sessions)
    cfg = state["config"]
    cfg.consolidation.calibrate_endpoint_enabled = calibrate_enabled
    cfg.paths.calibration_artifacts = tmp_path / "calibration_artifacts"
    cfg.paths.calibration_prompts = tmp_path / "calibration_prompts"
    cfg.paths.calibration_prompts.mkdir(parents=True, exist_ok=True)
    cfg.model_config = MagicMock()
    cfg.model_config.model_id = "test-model"

    state["model"] = MagicMock(name="model")
    state["tokenizer"] = MagicMock(name="tokenizer")
    state["memory_store"] = MagicMock(name="memory_store")
    state["router"] = MagicMock(name="router")
    store = state["speaker_store"]
    store.get_name.return_value = "Alex"
    store.resolve_speaker_name.return_value = "Alex"
    state["calibration_run"] = None
    state["store_quarantine"] = None
    state["pending_rehydration"] = False
    for key, value in overrides.items():
        state[key] = value
    return state


def _route_client(state, monkeypatch) -> "tuple[object, list[tuple[object, str]]]":
    """TestClient over the real app with *state* installed and the executor
    submission stubbed — the real arbitrator + real route boundary run;
    nothing is actually submitted to a thread pool.  Mirrors
    ``tests.server.test_consolidate_dispatch._route_client``.
    """
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


# ---------------------------------------------------------------------------
# Every /calibrate/* route: 200 {status, action[, run_id,
# artifact_dir]}, exactly one executor submission, no result body.
# ---------------------------------------------------------------------------


class TestEveryCalibrateRouteEnvelope:
    @pytest.mark.parametrize("route", _ALL_CALIBRATE_ROUTES)
    def test_started_calibration_carries_identity_and_submits_once(
        self, tmp_path, monkeypatch, route
    ) -> None:
        import paramem.server.app as app_module

        named = 1 if route == "extract_pending" else 0
        state = _make_calibrate_state(tmp_path, named_sessions=named)
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post(f"/calibrate/{route}", json=_payload_for(route))

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "started_calibration"
        expected_action = "calibrate_pending" if route == "extract_pending" else "calibrate"
        assert body["action"] == expected_action
        assert set(body) == {"status", "action", "run_id", "artifact_dir"}, (
            "no result body: the route returns identity only, never the run's payload"
        )
        assert body["run_id"]
        assert body["artifact_dir"]

        assert len(submitted) == 1, f"expected exactly one executor submission; got {submitted}"
        fn, status = submitted[0]
        assert status == "started_calibration"
        assert getattr(fn, "func", None) is app_module._run_calibration_sync
        spec = fn.args[0]
        assert spec.route_path == f"/calibrate/{route}"
        assert spec.run_id == body["run_id"]
        assert str(spec.artifact_dir) == body["artifact_dir"]


# ---------------------------------------------------------------------------
# Every arbitrator guard defers a calibrate call with 200
# deferred_*, never a raised exception / non-200.
# ---------------------------------------------------------------------------


class TestCalibrateRouteDefersOnEveryArbitratorGuard:
    def _assert_deferred(self, tmp_path, monkeypatch, route, state_patch, expected_status) -> None:
        state = _make_calibrate_state(tmp_path)
        for key, value in state_patch.items():
            state[key] = value
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post(f"/calibrate/{route}", json=_payload_for(route))

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == expected_status
        assert submitted == []

    def test_base_swap_active_defers(self, tmp_path, monkeypatch) -> None:
        self._assert_deferred(
            tmp_path,
            monkeypatch,
            "extract",
            {"migration": {"base_swap_active": True}},
            "deferred_base_swap_active",
        )

    def test_already_running_defers(self, tmp_path, monkeypatch) -> None:
        self._assert_deferred(
            tmp_path, monkeypatch, "extract", {"consolidating": True}, "deferred_already_running"
        )

    def test_cloud_only_defers(self, tmp_path, monkeypatch) -> None:
        self._assert_deferred(
            tmp_path, monkeypatch, "extract", {"mode": "cloud-only"}, "deferred_cloud_only"
        )

    def test_bg_training_defers(self, tmp_path, monkeypatch) -> None:
        bg = MagicMock()
        bg.is_training = True
        self._assert_deferred(
            tmp_path, monkeypatch, "extract", {"background_trainer": bg}, "deferred_bg_training"
        )

    def test_trial_active_defers(self, tmp_path, monkeypatch) -> None:
        self._assert_deferred(
            tmp_path,
            monkeypatch,
            "extract",
            {"migration": {"state": "TRIAL"}},
            "deferred_trial_active",
        )


# ---------------------------------------------------------------------------
# A calibrate run in flight makes /consolidate answer
# deferred_already_running, and vice versa: both sides borrow the same
# ``consolidating`` mutex the arbitrator's own guard reads.
# ---------------------------------------------------------------------------


class TestCalibrateAndConsolidationShareTheMutex:
    def test_calibrate_in_flight_defers_consolidate(self, tmp_path, monkeypatch) -> None:
        state = _make_calibrate_state(tmp_path, consolidating=True)
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/consolidate")

        assert resp.json() == {"status": "deferred_already_running", "action": "full"}
        assert submitted == []

    def test_consolidate_in_flight_defers_calibrate(self, tmp_path, monkeypatch) -> None:
        state = _make_calibrate_state(tmp_path, consolidating=True)
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.json()["status"] == "deferred_already_running"
        assert submitted == []


# ---------------------------------------------------------------------------
# A calibrate route answering started_migration returns no
# run_id/artifact_dir and leaves _state["calibration_run"] untouched, as do
# a deferred_* and a noop_*.  Only started_calibration writes it.
# ---------------------------------------------------------------------------


class TestIdentityWrittenOnlyOnStartedCalibration:
    def test_started_migration_carries_no_identity_and_leaves_record_untouched(
        self, tmp_path, monkeypatch
    ) -> None:
        state = _make_calibrate_state(tmp_path, pending_rehydration=True)
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "started_migration"
        assert set(body) == {"status", "action"}, "no identity on a pre-empted dispatch"
        assert state["calibration_run"] is None
        assert len(submitted) == 1  # the migration sync itself was submitted

    def test_deferred_leaves_record_untouched(self, tmp_path, monkeypatch) -> None:
        state = _make_calibrate_state(tmp_path, consolidating=True)
        state["calibration_run"] = {"run_id": "prior", "outcome": None}
        client, _submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.json()["status"] == "deferred_already_running"
        assert state["calibration_run"] == {"run_id": "prior", "outcome": None}

    def test_noop_leaves_record_untouched(self, tmp_path, monkeypatch) -> None:
        """``/calibrate/extract_pending`` with no pending session at all
        noops exactly as ``/consolidate/interim`` does, and writes no
        identity."""
        state = _make_calibrate_state(tmp_path)
        state["calibration_run"] = {"run_id": "prior", "outcome": None}
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract_pending", json={})

        assert resp.json()["status"].startswith("noop_")
        assert set(resp.json()) == {"status", "action"}
        assert state["calibration_run"] == {"run_id": "prior", "outcome": None}
        assert submitted == []


# ---------------------------------------------------------------------------
# Deferred_store_quarantined refuses a calibrate dispatch.
# ---------------------------------------------------------------------------


class TestStoreQuarantineRefusesCalibrate:
    def test_quarantined_store_defers_a_calibrate_route(self, tmp_path, monkeypatch) -> None:
        state = _make_calibrate_state(tmp_path, **{"store_quarantine": {"reason": "test"}})
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.json() == {"status": "deferred_store_quarantined", "action": "calibrate"}
        assert submitted == []


# ---------------------------------------------------------------------------
# Preflight refuses with 503 when model/tokenizer/memory_store is
# None, independently of _state["mode"].
# ---------------------------------------------------------------------------


class TestPreflightIndependentOfMode:
    def test_missing_model_503_even_though_mode_reads_local(self, tmp_path, monkeypatch) -> None:
        state = _make_calibrate_state(tmp_path)
        state["mode"] = "local"  # mode says local...
        state["model"] = None  # ...but the handle itself is absent
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.status_code == 503
        assert submitted == []

    def test_missing_memory_store_503_even_though_mode_reads_local(
        self, tmp_path, monkeypatch
    ) -> None:
        state = _make_calibrate_state(tmp_path)
        state["mode"] = "local"
        state["memory_store"] = None
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.status_code == 503
        assert submitted == []

    def test_flag_disabled_is_404_regardless_of_mode(self, tmp_path, monkeypatch) -> None:
        state = _make_calibrate_state(tmp_path, calibrate_enabled=False)
        client, submitted = _route_client(state, monkeypatch)

        resp = client.post("/calibrate/extract", json=_payload_for("extract"))

        assert resp.status_code == 404
        assert submitted == []
