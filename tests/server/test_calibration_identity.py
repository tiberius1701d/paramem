"""Run identity: ``run_stamp`` / ``artifact_run_dir`` / a run's own
``response.json`` landing in exactly the directory its 200 named, and the
``_state["calibration_run"]`` lifecycle across a real (non-executor-thread)
run.
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import paramem.server.app as app_module
import paramem.server.calibrate as calibrate_module
import paramem.utils.artifacts as artifacts_module
from paramem.utils.artifacts import artifact_run_dir, run_stamp

_ALL_CALIBRATE_ROUTE_PATHS = (
    "/calibrate/extract",
    "/calibrate/procedural",
    "/calibrate/anonymize",
    "/calibrate/enrich",
    "/calibrate/plausibility",
    "/calibrate/normalize",
    "/calibrate/anonymize_facts",
    "/calibrate/name",
    "/calibrate/respond",
    "/calibrate/extract_pending",
)


# ---------------------------------------------------------------------------
# Artifact_run_dir maps route path + stamp to the documented
# directory for all ten routes and for "campaigns".
# ---------------------------------------------------------------------------


class TestArtifactRunDirLayout:
    def test_run_stamp_matches_the_documented_utc_format(self) -> None:
        stamp = run_stamp()
        # UTC %Y%m%dT%H%M%SZ -- e.g. 20260101T120000Z
        assert re.fullmatch(r"\d{8}T\d{6}Z", stamp), stamp

    def test_every_calibrate_route_maps_to_calibrate_slash_stage_slash_stamp(self) -> None:
        root = Path("/root")
        stamp = "20260101T000000Z"
        for route_path in _ALL_CALIBRATE_ROUTE_PATHS:
            got = artifact_run_dir(root, route_path, stamp)
            stage = route_path.rsplit("/", 1)[-1]
            assert got == root / "calibrate" / stage / stamp, (route_path, got)

    def test_campaigns_maps_to_campaigns_slash_stamp(self) -> None:
        root = Path("/root")
        stamp = "20260101T000000Z"
        got = artifact_run_dir(root, "campaigns", stamp)
        assert got == root / "campaigns" / stamp

    def test_leading_and_trailing_slashes_are_normalized(self) -> None:
        root = Path("/root")
        stamp = "s"
        assert artifact_run_dir(root, "/calibrate/extract/", stamp) == artifact_run_dir(
            root, "calibrate/extract", stamp
        )

    def test_directory_is_not_created_by_the_mapping_alone(self, tmp_path: Path) -> None:
        """A path, not a promise the directory exists -- created on first
        write, never by artifact_run_dir itself."""
        got = artifact_run_dir(tmp_path, "/calibrate/extract", run_stamp())
        assert not got.exists()


# ---------------------------------------------------------------------------
# The run's run_id, its directory name, and every stamp inside it
# are equal; no int(time.time()) remains under the calibration root (AST
# scan of calibrate.py and artifacts.py).
# ---------------------------------------------------------------------------


def _calls_to_int_time_time(source: str) -> list[int]:
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and node.args[0].func.attr == "time"
            and isinstance(node.args[0].func.value, ast.Attribute)
            and node.args[0].func.value.attr == "time"
        ):
            hits.append(node.lineno)
    return hits


class TestNoAdHocTimestampUnderCalibrationRoot:
    def test_calibrate_py_has_no_int_time_time(self) -> None:
        source = inspect.getsource(calibrate_module)
        assert _calls_to_int_time_time(source) == []

    def test_artifacts_py_has_no_int_time_time(self) -> None:
        source = inspect.getsource(artifacts_module)
        assert _calls_to_int_time_time(source) == []


# ---------------------------------------------------------------------------
# A started_calibration response carries run_id and artifact_dir;
# the run writes response.json into exactly that directory.
# ---------------------------------------------------------------------------


class TestRunWritesResponseIntoItsOwnDirectory:
    def _minimal_spec(self, tmp_path: Path, *, run_id: str, artifact_dir: Path):
        params = calibrate_module.CalibrateParams()

        def _dispatch():
            return "raw output text", {"ok": True}

        return calibrate_module.CalibrationRunSpec(
            stage="extract",
            route_path="/calibrate/extract",
            run_id=run_id,
            artifact_dir=artifact_dir,
            dispatch=_dispatch,
            input_prompt_phase="local_extract",  # never opened -> unreached_step, harmless
            supports_seed=True,
            params=params,
            overrides={},
        )

    def test_response_json_lands_in_the_minted_artifact_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "calibration_artifacts"
        run_id = run_stamp()
        artifact_dir = artifact_run_dir(root, "/calibrate/extract", run_id)
        spec = self._minimal_spec(tmp_path, run_id=run_id, artifact_dir=artifact_dir)

        config = MagicMock()
        config.vram.cooldown_gate_threshold_c = 0  # disables the cooldown wait
        config.paths.data = tmp_path / "state"
        config.model_config.model_id = "test-model"

        state = {
            "config": config,
            "tokenizer": None,
            "calibration_run": {"run_id": run_id, "outcome": None},
            "event_loop": None,
            "mode": "local",
        }

        monkeypatch.setattr(app_module, "_state", state)
        app_module._run_calibration_sync(spec)

        response_path = artifact_dir / "response.json"
        assert response_path.exists(), f"expected {response_path} to exist"
        assert artifact_dir == root / "calibrate" / "extract" / run_id
        # No other response.json anywhere else under the calibration root.
        all_responses = list(root.rglob("response.json"))
        assert all_responses == [response_path]


# ---------------------------------------------------------------------------
# /status reports the latest started run and, after its terminal,
# its outcome.  StatusResponse.calibration_run is sourced verbatim from
# _state["calibration_run"] (grep-verified below); this test drives the
# underlying lifecycle through the real dispatch + executor-body functions
# rather than the full /status route (which needs an unrelated, large
# server-state fixture to build).
# ---------------------------------------------------------------------------


class TestCalibrationRunSlotLifecycle:
    def test_status_response_sources_calibration_run_from_state_verbatim(self) -> None:
        tree = ast.parse(inspect.getsource(app_module))
        found = False
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "StatusResponse"
            ):
                continue
            for kw in node.keywords:
                if kw.arg != "calibration_run":
                    continue
                for sub in ast.walk(kw.value):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "get"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "_state"
                        and sub.args
                        and isinstance(sub.args[0], ast.Constant)
                        and sub.args[0].value == "calibration_run"
                    ):
                        found = True
        assert found, "StatusResponse(calibration_run=...) must read _state.get('calibration_run')"

    def test_slot_holds_the_started_run_then_its_outcome_after_the_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_id = run_stamp()
        root = tmp_path / "calibration_artifacts"
        artifact_dir = artifact_run_dir(root, "/calibrate/extract", run_id)

        def _dispatch():
            from paramem.graph.phase_trace import phase_trace

            with phase_trace("local_extract"):
                pass
            return "raw output", {"ok": True}

        params = calibrate_module.CalibrateParams()
        spec = calibrate_module.CalibrationRunSpec(
            stage="extract",
            route_path="/calibrate/extract",
            run_id=run_id,
            artifact_dir=artifact_dir,
            dispatch=_dispatch,
            input_prompt_phase="local_extract",
            supports_seed=True,
            params=params,
            overrides={},
        )

        config = MagicMock()
        config.vram.cooldown_gate_threshold_c = 0
        config.paths.data = tmp_path / "state"
        config.model_config.model_id = "test-model"

        # The record a real _submit_calibration_run would have published at
        # dispatch time, BEFORE the run's own terminal fires.
        state = {
            "config": config,
            "tokenizer": None,
            "event_loop": None,
            "mode": "local",
            "calibration_run": {
                "run_id": run_id,
                "action": "calibrate",
                "route": "/calibrate/extract",
                "artifact_dir": str(artifact_dir),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "outcome": None,
                "finished_at": None,
            },
        }
        monkeypatch.setattr(app_module, "_state", state)

        assert state["calibration_run"]["outcome"] is None, "sanity: no outcome before the run"

        app_module._run_calibration_sync(spec)

        assert state["calibration_run"]["run_id"] == run_id
        assert state["calibration_run"]["outcome"] == "completed"
        assert state["calibration_run"]["finished_at"] is not None
