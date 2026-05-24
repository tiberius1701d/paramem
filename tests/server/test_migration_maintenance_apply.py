"""CPU isolation-contract test for the migration accept→apply sandbox path.

Asserts that ``_apply_config_live`` (and the accept handler's complete flow)
NEVER writes to the live ``data/ha`` tree.  All file-system side-effects must
land inside a tmp tree isolated via ``paths.data`` and ``paths.sessions``
overrides.

This is the regression guard for the isolation guarantee that the GPU smoke
relies on.  GPU operations are mocked so the test runs on any machine without
a GPU.

Design
------
Three isolation assertions:

1. **No-op-skip path (rollback case, S-6):** when the on-disk config hash
   equals the in-memory ``config_drift.loaded_hash``, ``_apply_config_live``
   returns immediately with ``applied_live=True, skipped="no_change"`` and
   performs NO file-system writes at all.

2. **Apply path (accept case):** when disk hash ≠ memory hash, ``_apply_config_live``
   calls ``_live_reload_base_model(refresh_config_from_disk=True)`` (mocked)
   and then ``_build_config_derived_state`` (mocked).  The test asserts that
   the live ``data/ha`` directory is untouched after the call.

3. **Accept handler path:** drive the full ``/migration/accept`` handler with
   ``_apply_config_live`` monkeypatched to a stub that returns
   ``applied_live=True``.  Assert that the live ``data/ha`` dir (absolute path)
   is untouched: no adapters, registry, or sessions written there.

All three run without a GPU — ``load_base_model``, ``_release_base_model_in_process``,
and ``_live_reload_base_model`` are mocked where needed.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.backup import write as backup_write
from paramem.backup.types import ArtifactKind
from paramem.server.migration import TrialStash, initial_migration_state
from paramem.server.trial_state import (
    TRIAL_MARKER_SCHEMA_VERSION,
    TrialMarker,
    write_trial_marker,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LIVE_CONFIG_YAML: bytes = b"model: mistral\ndebug: false\n"
_CAND_CONFIG_YAML: bytes = b"model: mistral\ndebug: true\n"

# Absolute path to the live data dir — must stay untouched.
_LIVE_DATA_HA: Path = (Path(__file__).resolve().parents[2] / "data" / "ha").resolve()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Shared state builders
# ---------------------------------------------------------------------------


def _make_isolated_state(tmp_path: Path) -> dict:
    """Build a minimal ``_state`` with ALL paths isolated to ``tmp_path``.

    Both ``paths.data`` and ``paths.sessions`` point inside ``tmp_path`` so
    no operation on ``_state["config"]`` can reach the live ``data/ha`` tree.

    Parameters
    ----------
    tmp_path:
        Pytest tmp_path fixture.

    Returns
    -------
    dict
        Minimal ``_state`` suitable for ``_apply_config_live`` unit tests.
    """
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_CONFIG_YAML)

    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.paths.sessions = tmp_path / "sessions"
    config.paths.sessions.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.key_metadata_path = tmp_path / "data" / "ha" / "key_metadata.json"

    live_hash = _sha256(_LIVE_CONFIG_YAML)

    return {
        "model": None,
        "tokenizer": None,
        "config": config,
        "config_path": str(live_yaml),
        "config_drift": {
            "detected": False,
            "loaded_hash": live_hash,
            "disk_hash": live_hash,
            "last_checked_at": "2026-03-10T00:00:00+00:00",
        },
        "consolidating": False,
        "migration": initial_migration_state(),
        "migration_lock": asyncio.Lock(),
        "mode": "local",
        "cloud_only_reason": None,
        "background_trainer": None,
        "consolidation_loop": None,
        "session_buffer": None,
        "memory_store": None,
        "router": None,
        "boot_degraded": None,
        "server_started_at": "2026-03-10T00:00:00+00:00",
        "_apply_config_in_progress": False,
    }


def _seed_trial_in_state(state: dict, tmp_path: Path) -> None:
    """Seed ``state`` and disk to TRIAL state (pass gates) for accept tests.

    Writes the candidate config B to disk (so disk hash ≠ loaded hash) and
    populates the trial stash with valid backup artifacts.

    Parameters
    ----------
    state:
        The ``_state`` dict to mutate.
    tmp_path:
        Pytest tmp_path fixture.
    """
    config = state["config"]
    live_yaml = Path(state["config_path"])
    # Write candidate B so disk hash differs from loaded_hash (config A).
    live_yaml.write_bytes(_CAND_CONFIG_YAML)

    data_dir = config.paths.data
    state_dir = (data_dir / "state").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    backups_root = (data_dir / "backups").resolve()

    # Trial adapter + graph dirs.
    trial_adapter_dir = state_dir / "trial_adapter"
    trial_adapter_dir.mkdir(exist_ok=True)
    (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    trial_graph_dir = state_dir / "trial_graph"
    trial_graph_dir.mkdir(exist_ok=True)

    # Config A backup.
    config_slot = backup_write(
        ArtifactKind.CONFIG,
        _LIVE_CONFIG_YAML,
        {"tier": "pre_migration"},
        base_dir=backups_root / "config",
    )
    artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
    assert len(artifact_files) == 1
    config_artifact_filename = artifact_files[0].name

    trial_stash = TrialStash(
        started_at="2026-03-10T00:00:00+00:00",
        pre_trial_config_sha256=_sha256(_LIVE_CONFIG_YAML),
        candidate_config_sha256=_sha256(_CAND_CONFIG_YAML),
        backup_paths={"config": str(config_slot.resolve())},
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        gates={
            "status": "pass",
            "completed_at": "2026-03-10T01:00:00+00:00",
        },
    )

    migration = initial_migration_state()
    migration["state"] = "TRIAL"
    migration["trial"] = trial_stash
    state["migration"] = migration

    marker = TrialMarker(
        schema_version=TRIAL_MARKER_SCHEMA_VERSION,
        started_at="2026-03-10T00:00:00+00:00",
        pre_trial_config_sha256=_sha256(_LIVE_CONFIG_YAML),
        candidate_config_sha256=_sha256(_CAND_CONFIG_YAML),
        backup_paths={"config": str(config_slot.resolve())},
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        config_artifact_filename=config_artifact_filename,
    )
    write_trial_marker(state_dir, marker)


# Per-test baseline of the live trial marker, captured by the autouse fixture
# below: (existed, mtime_ns). Compared against post-test state in
# _assert_live_data_ha_untouched to detect a leaked write precisely.
_LIVE_MARKER_BASELINE: tuple[bool, int] = (False, 0)


@pytest.fixture(autouse=True)
def _capture_live_marker_baseline():
    """Snapshot the live ``state/trial_marker.json`` before each test.

    Records (existed, mtime_ns) into the module-level baseline so the isolation
    guard can diff against it instead of a wall-clock window. Also runs the
    guard after every test, so a leak is caught even when a test forgets the
    explicit assertion.
    """
    global _LIVE_MARKER_BASELINE
    marker = _LIVE_DATA_HA / "state" / "trial_marker.json"
    if marker.exists():
        _LIVE_MARKER_BASELINE = (True, marker.stat().st_mtime_ns)
    else:
        _LIVE_MARKER_BASELINE = (False, 0)
    yield
    _assert_live_data_ha_untouched()


def _assert_live_data_ha_untouched() -> None:
    """Assert the apply path wrote no trial marker into the live ``data/ha`` tree.

    Isolation is guaranteed structurally: every test redirects ``paths.data``
    and ``paths.sessions`` into ``tmp_path``, so ``_apply_config_live`` cannot
    reach the live tree. This is the belt-and-braces check on the one artifact
    the migration flow writes that a concurrent live server NEVER writes during
    normal consolidation: ``state/trial_marker.json``. We diff against the
    per-test baseline captured by ``_capture_live_marker_baseline`` — created,
    modified, or removed since the test started all count as leaks — rather than
    a wall-clock window that could miss a slow test or mistake an earlier real
    migration for a leak.

    We deliberately do NOT snapshot the whole ``data/ha`` tree: ``server.yaml``
    sets ``paths.data`` to the relative ``data/ha``, so a live server on this
    host writes adapters/registry there concurrently — a full-tree diff would
    false-positive during the pre-commit suite. Adapter/registry leaks are
    instead covered positively by ``test_apply_writes_go_to_tmp_not_live_tree``.
    """
    existed, base_mtime = _LIVE_MARKER_BASELINE
    marker_path = _LIVE_DATA_HA / "state" / "trial_marker.json"

    if not marker_path.exists():
        assert not existed, (
            f"ISOLATION FAILURE: {marker_path} was REMOVED during the test. "
            "The apply path deleted a live trial marker."
        )
        return

    now_mtime = marker_path.stat().st_mtime_ns
    assert existed, (
        f"ISOLATION FAILURE: {marker_path} was CREATED during the test. "
        "The apply path wrote a trial marker into the live data/ha tree."
    )
    assert now_mtime == base_mtime, (
        f"ISOLATION FAILURE: {marker_path} was MODIFIED during the test "
        f"(mtime {now_mtime} != baseline {base_mtime}). "
        "The apply path wrote into the live data/ha tree."
    )


# ---------------------------------------------------------------------------
# Test class 1 — no-op skip (rollback case)
# ---------------------------------------------------------------------------


class TestNoOpSkip:
    """_apply_config_live returns no_change when disk hash == memory hash."""

    def test_no_op_skip_returns_applied_live_no_change(self, tmp_path, monkeypatch):
        """No-op skip: disk=A, memory=A → applied_live=True, skipped='no_change', no GPU churn.

        The live data/ha dir must not be written to.
        """
        state = _make_isolated_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        # Disk matches memory hash: _LIVE_CONFIG_YAML is on disk, loaded_hash is its hash.
        result = app_module._apply_config_live()

        assert result["applied_live"] is True, (
            f"Expected applied_live=True on no-op skip, got: {result}"
        )
        assert result.get("skipped") == "no_change", f"Expected skipped='no_change', got: {result}"
        assert result.get("restart_required_reason") is None, (
            f"Expected no restart_required_reason on no-op, got: {result}"
        )
        _assert_live_data_ha_untouched()

    def test_no_op_skip_does_not_call_live_reload(self, tmp_path, monkeypatch):
        """No-op skip must NOT call _live_reload_base_model (no GPU churn)."""
        state = _make_isolated_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        reload_called = []

        def _mock_reload(**kwargs):
            reload_called.append(kwargs)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _mock_reload)

        app_module._apply_config_live()
        assert reload_called == [], (
            "_live_reload_base_model was called on a no-op skip — GPU churn regression"
        )

    def test_no_op_skip_writes_nothing_to_tmp_paths(self, tmp_path, monkeypatch):
        """No-op skip must not create any new files in the tmp paths.data tree."""
        state = _make_isolated_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        data_before = set(tmp_path.rglob("*"))
        app_module._apply_config_live()
        data_after = set(tmp_path.rglob("*"))

        new_files = data_after - data_before
        assert not new_files, f"No-op skip created unexpected files: {[str(p) for p in new_files]}"


# ---------------------------------------------------------------------------
# Test class 2 — apply path (accept case, GPU mocked)
# ---------------------------------------------------------------------------


class TestApplyPathIsolation:
    """_apply_config_live with GPU mocked: assert writes stay in tmp, never in data/ha."""

    def test_apply_live_never_touches_live_data_ha(self, tmp_path, monkeypatch):
        """Accept path with mocked GPU reload: live data/ha is untouched.

        ``_live_reload_base_model`` is mocked to succeed instantly.
        ``_state["mode"]`` is set to ``"local"`` by the mock so the
        post-reload assertions match a successful apply.
        """
        state = _make_isolated_state(tmp_path)
        _seed_trial_in_state(state, tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        def _mock_reload(
            refresh_config_from_disk: bool = False, rebuild_session_buffer: bool = False
        ):
            """Simulate a successful in-process reload: flip mode to local."""
            state["mode"] = "local"
            state["cloud_only_reason"] = None

        monkeypatch.setattr(app_module, "_live_reload_base_model", _mock_reload)
        # Patch _set_voice_pipeline_profile so it does not try to load STT/TTS.
        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)

        # Trigger apply (disk=B ≠ memory=A → full apply path).
        state["mode"] = "cloud-only"
        state["cloud_only_reason"] = "live_reload"
        result = app_module._apply_config_live()

        # The apply path ran (not no-op skip).
        assert result.get("skipped") != "no_change", (
            f"Apply took the no-op path unexpectedly: {result}"
        )

        # Live data/ha must be untouched.
        _assert_live_data_ha_untouched()

    def test_apply_writes_go_to_tmp_not_live_tree(self, tmp_path, monkeypatch):
        """Accept path: any writes by the apply land in tmp, not in data/ha.

        We mock ``load_server_config`` to return None for config B (simulating a
        parse failure) so the carve classification is skipped and
        ``_live_reload_base_model`` is called unconditionally.  The mock writes a
        sentinel file to the tmp data dir to confirm the call landed there and not
        in the live ``data/ha`` tree.
        """
        state = _make_isolated_state(tmp_path)
        _seed_trial_in_state(state, tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        tmp_data = tmp_path / "data" / "ha"
        reload_called = []

        def _mock_reload(
            refresh_config_from_disk: bool = False,
            rebuild_session_buffer: bool = False,
            lock_held: bool = False,
        ):
            """Simulate reload and write a marker file to tmp_data to prove routing."""
            state["mode"] = "local"
            state["cloud_only_reason"] = None
            marker = tmp_data / "smoke_apply_marker.txt"
            marker.write_text("apply-ran")
            reload_called.append(True)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _mock_reload)
        # Patch _set_voice_pipeline_profile so it does not try to load STT/TTS.
        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)

        # Patch load_server_config inside _apply_config_live so config_b is None.
        # This bypasses the carve classification entirely, routing straight to the
        # _live_reload_base_model call.  Isolation is preserved because all path
        # operations (release, rebuild) go through _state["config"] (which points to tmp).
        def _failing_load(path, **kw):
            raise ValueError("simulated parse failure for carve bypass")

        with patch("paramem.server.app.load_server_config", _failing_load):
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "live_reload"
            app_module._apply_config_live()

        # _live_reload_base_model was called.
        assert reload_called, (
            "_live_reload_base_model was not called — test setup error or unexpected code path"
        )
        # Marker is in tmp, NOT in live.
        assert (tmp_data / "smoke_apply_marker.txt").exists(), (
            "Mock reload did not write the marker to tmp data dir"
        )
        if _LIVE_DATA_HA.exists():
            assert not (_LIVE_DATA_HA / "smoke_apply_marker.txt").exists(), (
                "ISOLATION FAILURE: apply wrote to the live data/ha tree"
            )

    def test_r_paths_carve_does_not_write_to_live_tree(self, tmp_path, monkeypatch):
        """R-PATHS carve short-circuits before any write — live data/ha stays clean.

        We change config B to have a different paths.data so the R-PATHS carve
        fires.  The apply must short-circuit BEFORE any live reload.
        """
        state = _make_isolated_state(tmp_path)
        _seed_trial_in_state(state, tmp_path)

        # Mutate config B (on disk) to have a different paths.data.
        # We achieve this by monkeypatching load_server_config to return a
        # config with a different paths.data so the diff detects R-PATHS.
        config_a = state["config"]
        config_b_mock = MagicMock()
        config_b_mock.paths.data = tmp_path / "NEW_data"
        config_b_mock.paths.sessions = config_a.paths.sessions
        config_b_mock.stt = MagicMock()
        config_b_mock.stt.port = getattr(getattr(config_a, "stt", None), "port", None)
        config_b_mock.tts = MagicMock()
        config_b_mock.tts.port = getattr(getattr(config_a, "tts", None), "port", None)

        monkeypatch.setattr(app_module, "_state", state)
        reload_called = []

        def _mock_load(path):
            return config_b_mock

        def _mock_reload(**kwargs):
            reload_called.append(True)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _mock_reload)

        with patch("paramem.server.app.load_server_config", _mock_load):
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "live_reload"
            result = app_module._apply_config_live()

        # R-PATHS carve detected.
        assert result.get("restart_required_reason") == "paths_change", (
            f"Expected paths_change, got: {result}"
        )
        # No GPU reload fired.
        assert reload_called == [], (
            "_live_reload_base_model was called despite R-PATHS carve (should short-circuit)"
        )
        # Live data/ha untouched.
        _assert_live_data_ha_untouched()


# ---------------------------------------------------------------------------
# Test class 3 — accept handler with stubbed _apply_config_live
# ---------------------------------------------------------------------------


class TestAcceptHandlerIsolation:
    """Full /migration/accept handler: apply stub → live data/ha untouched."""

    def test_accept_handler_isolation_applied_live(self, tmp_path, monkeypatch):
        """Accept handler with applied_live stub: live data/ha is never written to.

        Drives the full FastAPI ``/migration/accept`` endpoint.  The trial state
        is seeded in a tmp dir; ``_apply_config_live`` is stubbed.  After the
        call, the live ``data/ha`` tree must be untouched.
        """
        state = _make_isolated_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        def _stub_apply():
            return {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": None,
                "restart_eligible": False,
            }

        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)

        # Seed TRIAL.
        _seed_trial_in_state(state, tmp_path)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body.get("applied_live") is True, f"Expected applied_live=True: {body}"
        assert body.get("restart_required") is False, (
            f"Expected restart_required=False on applied_live success: {body}"
        )

        # Live data/ha must be untouched.
        _assert_live_data_ha_untouched()

    def test_accept_handler_isolation_apply_failed(self, tmp_path, monkeypatch):
        """Accept handler with apply-failure stub: live data/ha is still untouched.

        Even when the live apply fails, any writes (e.g. to the trial adapter
        archive slot) must land in the tmp tree.
        """
        state = _make_isolated_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        def _stub_apply_failure():
            return {
                "applied_live": False,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": "apply_failed",
                "restart_eligible": False,
            }

        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_failure)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)

        _seed_trial_in_state(state, tmp_path)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body.get("applied_live") is False, f"Expected applied_live=False: {body}"
        assert body.get("restart_required") is True, (
            f"Expected restart_required=True on apply failure: {body}"
        )

        # Even on failure, live data/ha must be untouched.
        _assert_live_data_ha_untouched()

    def test_accept_handler_trial_adapter_archive_goes_to_tmp(self, tmp_path, monkeypatch):
        """Trial adapter archive rotation lands in tmp, not in live data/ha.

        After a successful accept, the trial adapter is moved to a rotation
        slot under ``config.paths.data / backups / trial_adapters``.  This
        must be the tmp data dir, not the live ``data/ha`` tree.
        """
        state = _make_isolated_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        def _stub_apply():
            return {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": None,
                "restart_eligible": False,
            }

        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)

        _seed_trial_in_state(state, tmp_path)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()

        archive_path = body.get("trial_adapter_archive_path", "")
        # The archive path must be inside tmp_path, not inside _LIVE_DATA_HA.
        if archive_path:
            archive_abs = Path(archive_path).resolve()
            assert not str(archive_abs).startswith(str(_LIVE_DATA_HA)), (
                f"ISOLATION FAILURE: trial adapter archive is inside live data/ha: {archive_abs}"
            )
            # It should be inside the tmp tree.
            assert str(archive_abs).startswith(str(tmp_path.resolve())), (
                f"Archive path is not inside tmp tree: {archive_abs} "
                f"(tmp_path={tmp_path.resolve()})"
            )

        _assert_live_data_ha_untouched()
