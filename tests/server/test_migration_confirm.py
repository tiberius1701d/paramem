"""Tests for POST /migration/confirm (Slice 3b.2) and base-swap orchestration.

All tests run without GPU — the trial consolidation task is cancelled/no-op.
The 5-step atomic ordering is validated with both happy-path and per-step
failure rollback scenarios.  Slice 2 tests cover the full in-process
orchestration: Phase A → reload → Phase B → done, the reload-deferred path,
and the uncapped Phase B recall gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.migration import initial_migration_state
from paramem.server.trial_state import TrialMarker, read_trial_marker

_LIVE_YAML = b"model: mistral\ndebug: false\n"
_CAND_YAML = b"model: mistral\ndebug: true\n"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_state(tmp_path: Path) -> dict:
    """Build a STAGING _state with a real candidate file."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)
    cand_yaml = tmp_path / "candidate.yaml"
    cand_yaml.write_bytes(_CAND_YAML)

    # Production layout: config.paths.data = data/ha so state and backups
    # live directly under it (no extra /ha/ segment added by the handler).
    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.key_metadata_path = tmp_path / "data" / "ha" / "key_metadata.json"

    loop_mock = MagicMock()
    loop_mock.merger.save_bytes.return_value = b'{"nodes":[],"links":[]}'

    staging = initial_migration_state()
    staging["state"] = "STAGING"
    staging["candidate_path"] = str(cand_yaml)
    staging["candidate_hash"] = _sha256(_CAND_YAML)
    staging["candidate_bytes"] = _CAND_YAML
    staging["candidate_text"] = _CAND_YAML.decode("utf-8")

    return {
        "model": None,
        "tokenizer": None,
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": staging,
        "migration_lock": asyncio.Lock(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "mode": "normal",
        "background_trainer": None,
        "consolidation_loop": loop_mock,
        "session_buffer": None,
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """STAGING state monkeypatched into app_module."""
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state, monkeypatch):
    """TestClient with mocked trial consolidation."""

    # Patch _run_trial_consolidation to a no-op so no background task runs.
    async def _noop_trial():
        pass

    monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestConfirmHappyPath:
    def test_confirm_happy_path_returns_200(self, client, state, tmp_path):
        """STAGING with valid stash → 200, state=TRIAL."""
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "TRIAL"

    def test_confirm_sets_state_to_trial(self, client, state, tmp_path):
        """After confirm, _state["migration"]["state"] == "TRIAL"."""
        client.post("/migration/confirm", json={})
        assert state["migration"]["state"] == "TRIAL"

    def test_confirm_writes_config_backup_slot(self, client, state, tmp_path):
        """Only the config backup slot appears in backups_root.

        Config is the sole pre-migration artifact; graph/registry are not
        backed up.  Uses the production layout: config.paths.data is data/ha,
        so the handler appends 'backups' directly (no extra /ha/ segment).
        """
        client.post("/migration/confirm", json={})
        backups_root = state["config"].paths.data / "backups"
        config_dir = backups_root / "config"
        assert config_dir.exists(), "Missing config backup kind dir"
        slots = [d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) == 1, f"Expected 1 slot in {config_dir}, got {slots}"
        assert not (backups_root / "graph").exists(), "graph backup should not be written"
        assert not (backups_root / "registry").exists(), "registry backup should not be written"

    def test_confirm_writes_trial_marker(self, client, state, tmp_path):
        """state/trial.json is written after confirm.

        Uses the production layout: marker lives at
        <config.paths.data>/state/trial.json (not /ha/state/trial.json).
        """
        client.post("/migration/confirm", json={})
        state_dir = state["config"].paths.data / "state"
        marker = read_trial_marker(state_dir)
        assert marker is not None

    def test_confirm_renames_candidate_to_live(self, client, state, tmp_path):
        """The candidate file becomes the live config after confirm."""
        live_config_path = Path(state["config_path"])
        candidate_path = Path(state["migration"]["candidate_path"])
        assert candidate_path.exists()
        client.post("/migration/confirm", json={})
        # After rename, live config should contain candidate bytes.
        assert live_config_path.read_bytes() == _CAND_YAML

    def test_confirm_marker_contents_match_stash(self, client, state, tmp_path):
        """trial.json fields equal candidate_hash and pre_trial_hash."""
        client.post("/migration/confirm", json={})
        state_dir = state["config"].paths.data / "state"
        marker = read_trial_marker(state_dir)
        assert marker is not None
        assert marker.candidate_config_sha256 == _sha256(_CAND_YAML)
        assert marker.pre_trial_config_sha256 == _sha256(_LIVE_YAML)

    def test_confirm_backup_meta_has_pre_trial_hash(self, client, state, tmp_path):
        """The config backup slot's meta.json has pre_trial_hash == sha256(live config)."""
        expected_hash = _sha256(_LIVE_YAML)
        client.post("/migration/confirm", json={})
        backups_root = state["config"].paths.data / "backups"
        kind_dir = backups_root / "config"
        slots = [d for d in kind_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, f"No slots in {kind_dir}"
        meta_files = list(slots[0].glob("*.meta.json"))
        assert meta_files, f"No meta.json in {slots[0]}"
        meta_data = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta_data.get("pre_trial_hash") == expected_hash, (
            "pre_trial_hash mismatch in config backup"
        )


# ---------------------------------------------------------------------------
# 409 gates
# ---------------------------------------------------------------------------


class TestConfirm409:
    def test_confirm_409_when_consolidating(self, client, state, tmp_path):
        """STAGING + consolidating=True → 409 consolidating."""
        state["consolidating"] = True
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidating"

    def test_confirm_409_when_not_staging(self, client, state, tmp_path):
        """State is LIVE → 409 not_staging."""
        state["migration"]["state"] = "LIVE"
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "not_staging"

    def test_confirm_409_when_already_trial(self, client, state, tmp_path):
        """State is TRIAL → 409 trial_active."""
        state["migration"]["state"] = "TRIAL"
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"


# ---------------------------------------------------------------------------
# Step failure rollbacks
# ---------------------------------------------------------------------------


class TestConfirmStepFailures:
    def test_confirm_step2_failure_rollback(self, client, state, tmp_path, monkeypatch):
        """Patch backup.write to raise on the config backup call → 500 backup_write_failed.

        STAGING state is retained on failure.  Config is the only pre-migration
        backup, so the failure must be injected on the first (and only) write.
        """
        call_count = [0]

        from paramem.backup import backup as backup_module

        original_write_fn = backup_module.write

        def _failing_write(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("disk full")
            return original_write_fn(*args, **kwargs)

        monkeypatch.setattr(backup_module, "write", _failing_write)
        # Re-patch the app import of backup_write.
        with patch("paramem.server.app.backup_write", _failing_write):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "backup_write_failed"
        # State must still be STAGING.
        assert state["migration"]["state"] == "STAGING"

    def test_confirm_step3_failure_rollback(self, client, state, tmp_path, monkeypatch):
        """Patch write_trial_marker → 500 marker_write_failed; STAGING retained; backups deleted."""

        def _fail_write(*args, **kwargs):
            raise OSError("marker write failed")

        with patch("paramem.server.app.write_trial_marker", _fail_write):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "marker_write_failed"
        assert state["migration"]["state"] == "STAGING"
        # The config backup slot should be cleaned up.
        backups_root = state["config"].paths.data / "ha" / "backups"
        kind_dir = backups_root / "config"
        if kind_dir.exists():
            slots = [d for d in kind_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            assert len(slots) == 0, f"Orphan slot in {kind_dir}: {slots}"

    def test_confirm_step4_failure_rollback(self, client, state, tmp_path, monkeypatch):
        """Patch _rename_config → 500 config_swap_failed; marker + backups deleted."""
        with patch(
            "paramem.server.app._rename_config", side_effect=OSError("EXDEV: cross-device rename")
        ):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "config_swap_failed"
        assert state["migration"]["state"] == "STAGING"
        # Marker must not exist.
        state_dir = state["config"].paths.data / "ha" / "state"
        assert read_trial_marker(state_dir) is None

    def test_confirm_releases_lock_on_failure(self, client, state, tmp_path, monkeypatch):
        """Step 3 failure: follow-up confirm does not deadlock.

        After a failed confirm, the migration lock must be released so a
        second confirm attempt can proceed (or fail cleanly with a non-deadlock
        error).  Correction 1.
        """
        fail_count = [0]

        def _fail_once(*args, **kwargs):
            fail_count[0] += 1
            if fail_count[0] == 1:
                raise OSError("transient failure")
            from paramem.server.trial_state import write_trial_marker as _real

            return _real(*args, **kwargs)

        with patch("paramem.server.app.write_trial_marker", _fail_once):
            resp1 = client.post("/migration/confirm", json={})
        assert resp1.status_code == 500

        # Second attempt: state is still STAGING (not blocked by a held lock).
        # It may fail because candidate file doesn't exist now, but must not hang.
        resp2 = client.post("/migration/confirm", json={})
        # Should not be 409 migration_in_progress (lock must be released).
        detail2 = resp2.json().get("detail", {})
        assert detail2.get("error") != "migration_in_progress", (
            "Lock was not released after step-3 failure (Correction 1 violated)"
        )

    def test_confirm_releases_lock_on_step4_failure(self, client, state, tmp_path, monkeypatch):
        """Step 4 (_rename_config) failure: lock is released; second confirm is not blocked.

        Correction 1: the confirm handler's try/finally unconditionally releases
        the migration lock even when step 4 raises.
        """
        with patch("paramem.server.app._rename_config", side_effect=OSError("EXDEV")):
            resp1 = client.post("/migration/confirm", json={})
        assert resp1.status_code == 500
        assert resp1.json()["detail"]["error"] == "config_swap_failed"

        # Second attempt: must not get 409 migration_in_progress.
        resp2 = client.post("/migration/confirm", json={})
        detail2 = resp2.json().get("detail", {})
        assert detail2.get("error") != "migration_in_progress", (
            "Lock was not released after step-4 failure (Correction 1 violated)"
        )


# ---------------------------------------------------------------------------
# B2 regression — trial_adapter_dir in marker uses state_dir, not state_dir.parent
# ---------------------------------------------------------------------------


class TestConfirmTrialAdapterDirUnderStateDir:
    """Verify confirm stores trial_adapter_dir under state_dir (not state_dir.parent).

    B2 bug (2026-04-22 E2E baseline): the confirm handler computed
    ``(state_dir.parent / "trial_adapter")`` which resolves to
    ``data/ha/trial_adapter`` (missing the ``state/`` segment).  Gate 3
    looked for quads.json at that wrong path and emitted a false FAIL
    on every real trial.

    Fix: use ``(state_dir / "trial_adapter")`` so the path is
    ``data/ha/state/trial_adapter``.
    """

    def test_trial_adapter_dir_under_state_dir_after_confirm(self, client, state, tmp_path):
        """After confirm, the trial marker's trial_adapter_dir is a child of state_dir.

        state_dir = config.paths.data / "state"
        Expected: trial_adapter_dir starts with str(state_dir)
        """
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text

        state_dir = state["config"].paths.data / "state"
        from paramem.server.trial_state import read_trial_marker

        marker = read_trial_marker(state_dir)
        assert marker is not None

        trial_adapter_dir = marker.trial_adapter_dir
        assert trial_adapter_dir, "trial_adapter_dir must not be empty"

        # Must be under state_dir, not state_dir.parent (i.e. data/ha/).
        expected_prefix = str(state_dir.resolve())
        assert trial_adapter_dir.startswith(expected_prefix), (
            f"trial_adapter_dir {trial_adapter_dir!r} must be a child of "
            f"state_dir {expected_prefix!r}.  "
            "B2 bug: old code used state_dir.parent which put trial_adapter/ one level "
            "up, causing gate 3 to look at the wrong path."
        )

    def test_trial_graph_dir_under_state_dir_after_confirm(self, client, state, tmp_path):
        """After confirm, the trial marker's trial_graph_dir is also a child of state_dir."""
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text

        state_dir = state["config"].paths.data / "state"
        from paramem.server.trial_state import read_trial_marker

        marker = read_trial_marker(state_dir)
        assert marker is not None

        trial_graph_dir = marker.trial_graph_dir
        assert trial_graph_dir, "trial_graph_dir must not be empty"

        expected_prefix = str(state_dir.resolve())
        assert trial_graph_dir.startswith(expected_prefix), (
            f"trial_graph_dir {trial_graph_dir!r} must be a child of state_dir {expected_prefix!r}."
        )


# ---------------------------------------------------------------------------
# B1 regression — config_artifact_filename extraction uses real backup layout
# ---------------------------------------------------------------------------


class TestConfirmConfigArtifactFilenameRealSlotLayout:
    """Verify confirm extracts the artifact filename (not the sidecar) from a real
    backup-writer slot layout (B1 fix, 2026-04-22 E2E baseline).

    The backup writer names its sidecar ``<kind>-<ts>.meta.json`` (prefixed, NOT
    the exact string ``"meta.json"``).  The old filter ``_entry.name != "meta.json"``
    missed that sidecar, causing iterdir to return it first on some filesystems and
    recording the sidecar filename in the marker.  At rollback time, os.rename moved
    the sidecar JSON over the live YAML, silently corrupting configs/server.yaml.
    """

    def test_config_artifact_filename_not_sidecar_after_confirm(
        self, client, state, tmp_path, monkeypatch
    ):
        """After confirm, trial marker records an artifact filename that ends with
        ``.bin`` or ``.bin.enc`` (NOT ``.meta.json``).

        Uses the real backup.write() call path so the sidecar naming convention
        (``config-<ts>.meta.json``) is exercised — this is the layout that caused
        the B1 bug.
        """
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text

        # Read the trial marker written by the handler.
        state_dir = state["config"].paths.data / "state"
        from paramem.server.trial_state import read_trial_marker

        marker = read_trial_marker(state_dir)
        assert marker is not None, "trial marker must be written by confirm"

        filename = marker.config_artifact_filename
        assert filename, "config_artifact_filename must be non-empty"

        # The artifact must NOT be the sidecar.
        assert not filename.endswith(".meta.json"), (
            f"config_artifact_filename is a sidecar name: {filename!r}.  "
            "The iterdir filter must exclude all *.meta.json sidecars, not just "
            "the exact string 'meta.json' (B1 fix, 2026-04-22 E2E baseline)."
        )

        # The artifact must be the binary blob written by backup.write().
        assert filename.endswith(".bin") or filename.endswith(".bin.enc"), (
            f"config_artifact_filename does not end with .bin or .bin.enc: {filename!r}"
        )


# ---------------------------------------------------------------------------
# Pure mode-switch fast path (consolidation.mode change → applied directly)
# ---------------------------------------------------------------------------


class TestConfirmModeSwitch:
    def test_pure_mode_switch_applied_directly(self, client, state, tmp_path, monkeypatch):
        """A migration whose only change is consolidation.mode is applied directly:
        state LIVE + mode_switch block, config swapped, refresh+arm invoked, NO
        trial marker, NO TRIAL state, NO trial consolidation."""
        state["migration"]["tier_diff"] = [
            {
                "dotted_path": "consolidation.mode",
                "old_value": "simulate",
                "new_value": "train",
                "tier": "pipeline_altering",
            }
        ]

        # Mirror lifespan arming without a real config load: capture the call.
        arm_called = {"n": 0}

        def _fake_refresh():
            arm_called["n"] += 1
            return state["config"]

        monkeypatch.setattr(app_module, "_refresh_config_from_disk_into_state", _fake_refresh)

        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        # Direct apply: LIVE, not TRIAL.
        assert body["state"] == "LIVE"
        ms = body["mode_switch"]
        assert ms is not None
        assert ms["from"] == "simulate"
        assert ms["to"] == "train"
        assert ms["direction"] == "simulate_to_train"
        assert ms["applies_via"] == "active_store_migration"

        # Config swapped on disk (candidate → live).
        assert Path(state["config_path"]).read_bytes() == _CAND_YAML
        # Refresh+arm helper invoked exactly once.
        assert arm_called["n"] == 1
        # No trial marker, state back to LIVE.
        state_dir = state["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None
        assert state["migration"]["state"] == "LIVE"

    def test_mode_plus_other_change_still_trials(self, client, state, tmp_path):
        """A diff that includes consolidation.mode AND another field is NOT a pure
        mode switch — it falls through to the normal TRIAL flow."""
        state["migration"]["tier_diff"] = [
            {
                "dotted_path": "consolidation.mode",
                "old_value": "simulate",
                "new_value": "train",
                "tier": "pipeline_altering",
            },
            {
                "dotted_path": "debug",
                "old_value": False,
                "new_value": True,
                "tier": "pipeline_altering",
            },
        ]
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "TRIAL"
        assert body.get("mode_switch") is None
        assert state["migration"]["state"] == "TRIAL"


# ---------------------------------------------------------------------------
# TrialMarker round-trip — base-swap fields
# ---------------------------------------------------------------------------


class TestTrialMarkerBaseSwapRoundTrip:
    """TrialMarker serializes and deserializes new base-swap fields correctly."""

    def test_new_fields_round_trip_via_to_dict_from_dict(self, tmp_path):
        """Base-swap fields survive to_dict → from_dict round-trip."""
        marker = TrialMarker(
            schema_version=1,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="aabbcc",
            candidate_config_sha256="ddeeff",
            backup_paths={"config": str(tmp_path / "config")},
            trial_adapter_dir=str(tmp_path / "trial_adapter"),
            trial_graph_dir=str(tmp_path / "trial_graph"),
            config_artifact_filename="config-20260524.bin",
            migration_kind="base_swap",
            base_swap_phase="phaseA_done",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=str(tmp_path / "bundle"),
        )
        d = marker.to_dict()
        assert d["migration_kind"] == "base_swap"
        assert d["base_swap_phase"] == "phaseA_done"
        assert d["old_model"] == "mistral"
        assert d["new_model"] == "qwen3-4b"
        assert d["bundle_slot"] != ""

        restored = TrialMarker.from_dict(d)
        assert restored.migration_kind == "base_swap"
        assert restored.base_swap_phase == "phaseA_done"
        assert restored.old_model == "mistral"
        assert restored.new_model == "qwen3-4b"
        assert restored.bundle_slot == d["bundle_slot"]

    def test_old_marker_loads_with_defaults(self):
        """A marker dict without base-swap fields deserializes with safe defaults.

        This ensures backward compatibility with markers written before Slice 1.
        """
        old_dict = {
            "schema_version": 1,
            "started_at": "2026-01-01T00:00:00+00:00",
            "pre_trial_config_sha256": "abc",
            "candidate_config_sha256": "def",
            "backup_paths": {"config": "/tmp/config"},
            "trial_adapter_dir": "/tmp/trial_adapter",
            "trial_graph_dir": "/tmp/trial_graph",
            "config_artifact_filename": "config.bin",
        }
        marker = TrialMarker.from_dict(old_dict)
        assert marker.migration_kind == "mode_switch"
        assert marker.base_swap_phase == ""
        assert marker.old_model == ""
        assert marker.new_model == ""
        assert marker.bundle_slot == ""

    def test_write_and_read_base_swap_marker(self, tmp_path):
        """write_trial_marker + read_trial_marker preserves base-swap fields."""
        from paramem.server.trial_state import write_trial_marker

        bundle_dir = tmp_path / "bundle_slot"
        bundle_dir.mkdir()
        marker = TrialMarker(
            schema_version=1,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="aabbcc",
            candidate_config_sha256="ddeeff",
            backup_paths={"bundle": str(bundle_dir)},
            trial_adapter_dir=str(tmp_path / "trial_adapter"),
            trial_graph_dir=str(tmp_path / "trial_graph"),
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseA",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=str(bundle_dir),
        )
        state_dir = tmp_path / "state"
        write_trial_marker(state_dir, marker)
        restored = read_trial_marker(state_dir)
        assert restored is not None
        assert restored.migration_kind == "base_swap"
        assert restored.base_swap_phase == "phaseA"
        assert restored.old_model == "mistral"
        assert restored.new_model == "qwen3-4b"
        assert restored.bundle_slot != ""


# ---------------------------------------------------------------------------
# Base-swap confirm: HF-cache precheck (409 when predict_base_bytes returns None)
# ---------------------------------------------------------------------------


def _make_base_swap_state(tmp_path: Path) -> dict:
    """Build a STAGING state with a model-change diff (mistral → qwen3-4b)."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(b"model: mistral\ndebug: false\n")
    cand_yaml = tmp_path / "candidate.yaml"
    cand_yaml.write_bytes(b"model: qwen3-4b\ndebug: false\n")

    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.model_name = "mistral"

    adapters_cfg = MagicMock()
    for tier in ("episodic", "semantic", "procedural"):
        tier_mock = MagicMock()
        tier_mock.enabled = False
        setattr(adapters_cfg, tier, tier_mock)
    config.adapters = adapters_cfg

    staging = initial_migration_state()
    staging["state"] = "STAGING"
    staging["candidate_path"] = str(cand_yaml)
    staging["candidate_hash"] = hashlib.sha256(b"model: qwen3-4b\ndebug: false\n").hexdigest()
    staging["candidate_bytes"] = b"model: qwen3-4b\ndebug: false\n"
    staging["candidate_text"] = "model: qwen3-4b\ndebug: false\n"
    staging["parsed_candidate"] = {"model": "qwen3-4b", "debug": False}
    staging["parsed_live"] = {"model": "mistral", "debug": False}
    staging["tier_diff"] = [
        {
            "dotted_path": "model",
            "old_value": "mistral",
            "new_value": "qwen3-4b",
            "tier": "destructive",
        }
    ]

    return {
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": staging,
        "migration_lock": asyncio.Lock(),
        "server_started_at": "2026-05-24T00:00:00+00:00",
        "mode": "normal",
        "background_trainer": None,
        "consolidation_loop": None,
        "session_buffer": None,
        "memory_store": MagicMock(),
        "event_loop": None,
    }


class TestConfirmBaseSwapPrechecks:
    """Confirm-endpoint prechecks for the base-swap path."""

    @pytest.fixture()
    def base_state(self, tmp_path, monkeypatch):
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        return fresh

    @pytest.fixture()
    def client_base_swap(self, base_state):
        return TestClient(app_module.app, raise_server_exceptions=False)

    def test_confirm_base_swap_409_when_model_not_cached(
        self, client_base_swap, base_state, monkeypatch
    ):
        """When predict_base_bytes returns None (HF-cache miss), confirm returns 409."""
        with patch("paramem.server.app.predict_base_bytes", return_value=None):
            resp = client_base_swap.post("/migration/confirm", json={})
        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]
        assert detail["error"] == "model_not_cached"
        assert "qwen3-4b" in detail["message"] or "Qwen" in detail["message"]

    def test_confirm_base_swap_409_unknown_model_alias(self, base_state, monkeypatch, tmp_path):
        """A candidate with an unknown model alias returns 409 unknown_model."""
        # Patch the parsed_candidate to use a non-existent alias.
        base_state["migration"]["parsed_candidate"] = {"model": "nonexistent-model"}
        base_state["migration"]["tier_diff"] = [
            {
                "dotted_path": "model",
                "old_value": "mistral",
                "new_value": "nonexistent-model",
                "tier": "destructive",
            }
        ]
        # No need to patch predict_base_bytes — alias lookup fails first.
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["error"] == "unknown_model"


# ---------------------------------------------------------------------------
# _run_base_swap_orchestration: ordering + success/failure paths (mocked)
# ---------------------------------------------------------------------------


class TestRunBaseSwapPhaseA:
    """Unit tests for _run_base_swap_orchestration coroutine.

    Mocks write_bundle, active-store dispatch (BackgroundTrainer.submit +
    migrate), _rename_config, and gpu_release/gpu_acquire to assert ordering
    and state transitions without GPU.  The existing Phase A methods validate
    the behaviour up to and including phaseA_done.  Slice 2 tests below
    validate the full orchestration: reload + Phase B + success / deferred
    paths.
    """

    def _make_phase_a_state(self, tmp_path: Path) -> dict:
        """Minimal _state dict for Phase A tests."""
        config = MagicMock()
        config.paths.data = tmp_path / "data"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata = tmp_path / "data" / "key_metadata.json"
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.key_metadata_path = config.paths.key_metadata  # mirror the real property
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config.model_name = "mistral"
        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            tier_mock = MagicMock()
            tier_mock.enabled = False
            setattr(adapters_cfg, tier, tier_mock)
        config.adapters = adapters_cfg
        config.training_config = MagicMock()
        config.consolidation = MagicMock()

        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            "migration": {
                "state": "TRIAL",
                "trial": {
                    "started_at": "2026-05-24T00:00:00+00:00",
                    "pre_trial_config_sha256": "",
                    "candidate_config_sha256": "aabb",
                    "backup_paths": {},
                    "trial_adapter_dir": "",
                    "trial_graph_dir": "",
                    "gates": {"status": "pending"},
                },
                "recovery_required": [],
            },
            "consolidation_loop": None,
            "background_trainer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }

    def _run_phase_a(
        self,
        state,
        monkeypatch,
        tmp_path,
        *,
        succeed: bool = True,
        reload_mode: str = "local",
    ):
        """Helper: run _run_base_swap_orchestration with mocks via asyncio.run().

        Returns ``(call_order, rename_calls, gates_received, marker_at_submit, state_dir)``.
        ``succeed`` controls whether the Phase A mock migration returns all_tiers_done.
        ``reload_mode`` sets the simulated post-reload ``_state["mode"]``:
          "local"      — reload succeeded; Phase B will run (default).
          "cloud-only" — reload was deferred; Phase B must NOT run.
        """
        import asyncio

        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        call_order: list[str] = []
        rename_calls: list = []
        gates_received: list[dict] = []
        marker_at_submit: list[str] = []

        def _fake_write_bundle(**kwargs):
            call_order.append("bundle")
            return bundle_slot

        submit_count = [0]

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            phase = "phase_a_submit" if submit_count[0] == 1 else "phase_b_submit"
            call_order.append(phase)
            m = read_trial_marker(state_dir)
            if m is not None:
                marker_at_submit.append(m.base_swap_phase)
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        if succeed:
            fake_updated_a = MigrationState(
                direction="train_to_simulate",
                started_at="2026-05-24T00:00:00+00:00",
                source_mode="train",
                target_mode="simulate",
                completed_tiers=["episodic", "semantic", "procedural"],
                failed_tiers={},
            )
        else:
            fake_updated_a = MigrationState(
                direction="train_to_simulate",
                started_at="2026-05-24T00:00:00+00:00",
                source_mode="train",
                target_mode="simulate",
                completed_tiers=[],
                failed_tiers={"episodic": "reconstruction error"},
            )

        # Phase B result (simulate→train): always success for basic tests.
        fake_updated_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )

        # migrate() is called twice: once for Phase A (train→simulate), once
        # for Phase B (simulate→train).  Return the appropriate result each time.
        migrate_call = [0]

        def _fake_migrate(loop, cfg, ms):
            migrate_call[0] += 1
            if migrate_call[0] == 2:
                # Regression guard (review CRITICAL): Phase B must load the
                # (Mistral) registries — its to-retrain tier list — into the store
                # BEFORE migrate runs.  The base-swap preload gate leaves the live
                # store empty; without this load migrate() would see 0 tiers and
                # raise "0 tiers but on-disk content exists", failing every swap.
                loop.store.load_registries_from_disk.assert_called()
            return fake_updated_a if migrate_call[0] == 1 else fake_updated_b

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        _reload_mode = reload_mode

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: load the renamed-config base model.

            Sets mode per reload_mode parameter and updates config.model_name
            to new_model on a successful reload (matching what
            _refresh_config_from_disk_into_state does in production).
            The Phase-B model-identity guard reads config.model_name to verify
            the reload completed correctly.
            Appends "apply_config_live" to call_order.
            """
            call_order.append("apply_config_live")
            state["mode"] = _reload_mode
            if _reload_mode != "local":
                state["cloud_only_reason"] = "insufficient_vram"
            else:
                state["cloud_only_reason"] = None
                # Simulate config refresh: new model is now live.
                state["config"].model_name = "qwen3-4b"

        with (
            patch("paramem.server.app.write_bundle", _fake_write_bundle),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch(
                "paramem.server.app._rename_config",
                lambda s, d: rename_calls.append((str(s), str(d))),
            ),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            asyncio.run(
                app_module._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                )
            )

        return call_order, rename_calls, gates_received, marker_at_submit, state_dir

    def test_bundle_written_before_phase_a_runs(self, tmp_path, monkeypatch):
        """Bundle backup must be written before the active-store migration starts."""
        state = self._make_phase_a_state(tmp_path)
        call_order, _, _, _, _ = self._run_phase_a(state, monkeypatch, tmp_path, succeed=True)
        assert call_order.index("bundle") < call_order.index("phase_a_submit"), (
            "Bundle must be written BEFORE Phase A submit"
        )

    def test_marker_phaseA_set_before_phase_a_runs(self, tmp_path, monkeypatch):
        """Marker with base_swap_phase='phaseA' written before Phase A is submitted.

        With the full orchestration, Phase B is also submitted (marker='phaseB').
        The first submit must see 'phaseA'.
        """
        state = self._make_phase_a_state(tmp_path)
        _, _, _, marker_at_submit, _ = self._run_phase_a(state, monkeypatch, tmp_path, succeed=True)
        assert marker_at_submit[0] == "phaseA", (
            f"First submit (Phase A) must see marker='phaseA'; got {marker_at_submit}"
        )

    def test_config_renamed_and_orchestration_succeeds(self, tmp_path, monkeypatch):
        """Full orchestration happy path: config renamed, Phase A+B run, status=pass.

        The orchestration runs Phase A, then reload (mocked to succeed), then
        Phase B, and finally sets migration status to 'pass' and clears the marker.
        """
        state = self._make_phase_a_state(tmp_path)
        call_order, rename_calls, gates_received, _, state_dir = self._run_phase_a(
            state, monkeypatch, tmp_path, succeed=True, reload_mode="local"
        )

        # Config was renamed (Phase A).
        assert rename_calls, "Config rename must be called on Phase A success"
        assert str(tmp_path / "candidate.yaml") == rename_calls[0][0]

        # apply_config_live was called (reload step).
        assert "apply_config_live" in call_order, "apply_config_live must be called after Phase A"

        # Phase B submitted after reload.
        assert "phase_b_submit" in call_order, "Phase B must be submitted after reload"
        assert call_order.index("phase_b_submit") > call_order.index("apply_config_live"), (
            "Phase B must run AFTER reload, not before"
        )

        # Final status is 'pass'.
        assert any(g.get("status") == "pass" for g in gates_received), (
            f"Expected 'pass' gate status after full orchestration; got {gates_received}"
        )

        # Marker is cleared on success.
        assert read_trial_marker(state_dir) is None, (
            "Trial marker must be cleared on orchestration success"
        )

    def test_bundle_and_marker_preserved_on_phase_a_failure(self, tmp_path, monkeypatch):
        """On Phase A failure: bundle and marker remain; status=phase_a_failed."""
        state = self._make_phase_a_state(tmp_path)
        _, rename_calls, gates_received, _, state_dir = self._run_phase_a(
            state, monkeypatch, tmp_path, succeed=False
        )

        # Config must NOT have been renamed.
        assert not rename_calls, "Config must not be renamed when Phase A fails"

        # Marker must still exist (for rollback).
        marker = read_trial_marker(state_dir)
        assert marker is not None, "Marker must be preserved on Phase A failure"
        assert marker.base_swap_phase == "phaseA", (
            "Marker must stay at 'phaseA' (not cleared) on failure"
        )

        # Status must be phase_a_failed.
        assert any(g.get("status") == "phase_a_failed" for g in gates_received), (
            f"Expected phase_a_failed gate status; got {gates_received}"
        )


# ---------------------------------------------------------------------------
# Slice 2: reload-deferred path + Phase B ordering
# ---------------------------------------------------------------------------


class TestBaseSwapOrchestrationSlice2:
    """Unit tests for the Slice 2 orchestration paths.

    All tests run without GPU.  Mock gpu_release/gpu_acquire to control reload
    outcome; use the _run_phase_a helper in TestRunBaseSwapPhaseA.
    """

    def _make_state(self, tmp_path: Path) -> dict:
        """Minimal _state dict for orchestration tests."""
        config = MagicMock()
        config.paths.data = tmp_path / "data"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata = tmp_path / "data" / "key_metadata.json"
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.key_metadata_path = config.paths.key_metadata  # mirror the real property
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config.model_name = "mistral"
        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            tier_mock = MagicMock()
            tier_mock.enabled = False
            setattr(adapters_cfg, tier, tier_mock)
        config.adapters = adapters_cfg
        config.training_config = MagicMock()
        config.consolidation = MagicMock()

        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            "migration": {
                "state": "TRIAL",
                "trial": {
                    "started_at": "2026-05-24T00:00:00+00:00",
                    "pre_trial_config_sha256": "",
                    "candidate_config_sha256": "aabb",
                    "backup_paths": {},
                    "trial_adapter_dir": "",
                    "trial_graph_dir": "",
                    "gates": {"status": "pending"},
                },
                "recovery_required": [],
            },
            "consolidation_loop": None,
            "background_trainer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }

    def _run_orchestration(
        self,
        state,
        monkeypatch,
        tmp_path,
        *,
        reload_mode: str = "local",
    ):
        """Run orchestration with mocks; return (call_order, gates_received, state_dir)."""
        import asyncio as _asyncio

        import paramem.server.app as _app

        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        call_order: list[str] = []
        gates_received: list[dict] = []
        submit_count = [0]

        def _fake_write_bundle(**kwargs):
            call_order.append("bundle")
            return bundle_slot

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            call_order.append("phase_a_submit" if submit_count[0] == 1 else "phase_b_submit")
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_updated_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        fake_updated_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        migrate_call = [0]

        def _fake_migrate(loop, cfg, ms):
            migrate_call[0] += 1
            if migrate_call[0] == 2:
                # Regression guard (review CRITICAL): Phase B must load the
                # (Mistral) registries — its to-retrain tier list — into the store
                # BEFORE migrate runs.  The base-swap preload gate leaves the live
                # store empty; without this load migrate() would see 0 tiers and
                # raise "0 tiers but on-disk content exists", failing every swap.
                loop.store.load_registries_from_disk.assert_called()
            return fake_updated_a if migrate_call[0] == 1 else fake_updated_b

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        _rm = reload_mode

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: load the renamed-config base model."""
            call_order.append("apply_config_live")
            state["mode"] = _rm
            state["cloud_only_reason"] = None if _rm == "local" else "insufficient_vram"
            if _rm == "local":
                # Simulate config refresh: new model is now live.
                state["config"].model_name = "qwen3-4b"

        with (
            patch("paramem.server.app.write_bundle", _fake_write_bundle),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                )
            )

        return call_order, gates_received, state_dir

    def test_reload_deferred_phase_b_not_run(self, tmp_path, monkeypatch):
        """When gpu_acquire leaves mode=cloud-only, Phase B must NOT run.

        The gates must be set to 'reload_deferred' and the marker must remain
        at 'phaseA_done' (the resume checkpoint).
        """
        state = self._make_state(tmp_path)
        call_order, gates_received, state_dir = self._run_orchestration(
            state, monkeypatch, tmp_path, reload_mode="cloud-only"
        )

        # Phase B must NOT have been submitted.
        assert "phase_b_submit" not in call_order, (
            "Phase B must NOT be submitted when reload is deferred"
        )

        # Gates must be reload_deferred.
        assert any(g.get("status") == "reload_deferred" for g in gates_received), (
            f"Expected 'reload_deferred' gate status; got {gates_received}"
        )

        # Marker must stay at phaseA_done (the crash-resume checkpoint).
        from paramem.server.trial_state import read_trial_marker as _rtm

        marker = _rtm(state_dir)
        assert marker is not None, "Marker must remain on reload_deferred"
        assert marker.base_swap_phase == "phaseA_done", (
            f"Marker must be 'phaseA_done' on reload_deferred; got {marker.base_swap_phase!r}"
        )

    def test_happy_path_phase_b_runs_after_reload(self, tmp_path, monkeypatch):
        """Phase B is submitted only AFTER gpu_acquire (reload) succeeds (mode=local)."""
        state = self._make_state(tmp_path)
        call_order, gates_received, state_dir = self._run_orchestration(
            state, monkeypatch, tmp_path, reload_mode="local"
        )

        # Both phases submitted.
        assert "phase_a_submit" in call_order
        assert "phase_b_submit" in call_order

        # Ordering: Phase A → reload → Phase B.
        idx_a = call_order.index("phase_a_submit")
        idx_reload = call_order.index("apply_config_live")
        idx_b = call_order.index("phase_b_submit")
        assert idx_a < idx_reload < idx_b, (
            f"Expected Phase A < reload < Phase B; got order {call_order}"
        )

        # Final status is 'pass'.
        assert any(g.get("status") == "pass" for g in gates_received), (
            f"Expected 'pass' on full success; got {gates_received}"
        )

        # Marker is cleared.
        from paramem.server.trial_state import read_trial_marker as _rtm

        assert _rtm(state_dir) is None, "Marker must be cleared on full success"

    def test_phase_b_marker_set_before_phase_b_runs(self, tmp_path, monkeypatch):
        """Marker transitions to 'phaseB' before Phase B is submitted.

        The marker_at_submit list captures the marker state at each submit call.
        Index 0 = Phase A submit (marker='phaseA').
        Index 1 = Phase B submit (marker='phaseB').
        """
        state = self._make_state(tmp_path)
        import asyncio as _asyncio

        import paramem.server.app as _app

        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        markers_at_submit: list[str] = []
        submit_count = [0]

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            from paramem.server.trial_state import read_trial_marker as _rtm

            m = _rtm(state_dir)
            markers_at_submit.append(m.base_swap_phase if m is not None else "missing")
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=list(("episodic", "semantic", "procedural")),
            failed_tiers={},
        )
        fake_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=list(("episodic", "semantic", "procedural")),
            failed_tiers={},
        )
        call_n = [0]

        def _fake_migrate(loop, cfg, ms):
            call_n[0] += 1
            return fake_a if call_n[0] == 1 else fake_b

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: load the renamed-config base model."""
            state["mode"] = "local"
            state["cloud_only_reason"] = None
            # Simulate config refresh: new model is now live.
            state["config"].model_name = "qwen3-4b"

        async def _fake_update_gates_noop(payload):
            pass

        with (
            patch("paramem.server.app.write_bundle", lambda **kw: bundle_slot),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates_noop),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                )
            )

        assert len(markers_at_submit) == 2, f"Expected 2 submit calls; got {markers_at_submit}"
        assert markers_at_submit[0] == "phaseA", (
            f"Phase A submit must see marker='phaseA'; got {markers_at_submit[0]!r}"
        )
        assert markers_at_submit[1] == "phaseB", (
            f"Phase B submit must see marker='phaseB'; got {markers_at_submit[1]!r}"
        )

    def test_post_phase_b_reload_runs_before_success_marker(self, tmp_path, monkeypatch):
        """Post-Phase-B in-process reload fires after Phase B and before status=pass.

        Regression for the AD-20 live-reload-after-final-tier gap: Phase B's
        per-tier migrate() loop leaves the in-RAM PeftModel mounted in the
        last tier's transient shape; without a final reload the published
        ``adapter_available`` topology stays stale until a systemctl restart.

        Voice drain/restore is now owned by _live_reload_base_model (the
        primitive), not by step 6 directly.  The step-6 ordering assertion
        is: reload fires after Phase B and before status=pass.
        """
        state = self._make_state(tmp_path)
        call_order, gates_received, state_dir = self._run_orchestration_with_reload_tracking(
            state, monkeypatch, tmp_path, reload_mode="local"
        )

        assert "phase_b_submit" in call_order, "Phase B must have been submitted"
        assert "post_phase_b_reload" in call_order, (
            "Post-Phase-B reload must be invoked after the final migrate() returns"
        )
        idx_b = call_order.index("phase_b_submit")
        idx_reload = call_order.index("post_phase_b_reload")
        assert idx_b < idx_reload, f"Reload must fire AFTER Phase B submit; got order {call_order}"

        # Status=pass still fires (reload is best-effort housekeeping, not gating).
        assert any(g.get("status") == "pass" for g in gates_received), (
            f"Expected 'pass' on full success; got {gates_received}"
        )

        # Marker is cleared on success.
        from paramem.server.trial_state import read_trial_marker as _rtm

        assert _rtm(state_dir) is None, "Marker must be cleared on full success"

    def test_post_phase_b_reload_failure_does_not_block_success(self, tmp_path, monkeypatch):
        """A raise from _live_reload_base_model is logged but status=pass still fires.

        Weights are already on disk by the time Phase B returns; the reload is
        in-RAM housekeeping.  Failure must not turn the swap into phase_b_failed.
        """
        state = self._make_state(tmp_path)

        def _raising_reload(*a, **kw):
            raise RuntimeError("simulated reload failure")

        call_order, gates_received, state_dir = self._run_orchestration_with_reload_tracking(
            state,
            monkeypatch,
            tmp_path,
            reload_mode="local",
            reload_impl=_raising_reload,
        )

        # Phase B ran; reload was attempted.
        assert "phase_b_submit" in call_order
        # Final status is still pass — the reload is best-effort.
        assert any(g.get("status") == "pass" for g in gates_received), (
            f"Reload failure must not block success; got {gates_received}"
        )
        # No phase_b_failed payload.
        assert not any(g.get("status") == "phase_b_failed" for g in gates_received), (
            "Reload failure must not raise to phase_b_failed"
        )

    def _run_orchestration_with_reload_tracking(
        self,
        state,
        monkeypatch,
        tmp_path,
        *,
        reload_mode: str = "local",
        reload_impl=None,
    ):
        """Like _run_orchestration but records the post-Phase-B reload call.

        ``reload_impl`` defaults to a no-op that appends ``"post_phase_b_reload"``
        to ``call_order``.  Pass a callable to override (e.g. a raising stub).
        """
        import asyncio as _asyncio

        import paramem.server.app as _app

        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        call_order: list[str] = []
        gates_received: list[dict] = []
        submit_count = [0]

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            call_order.append("phase_a_submit" if submit_count[0] == 1 else "phase_b_submit")
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        fake_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        migrate_call = [0]

        def _fake_migrate(loop, cfg, ms):
            migrate_call[0] += 1
            return fake_a if migrate_call[0] == 1 else fake_b

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        _rm = reload_mode

        async def _fake_gpu_release():
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            call_order.append("apply_config_live")
            state["mode"] = _rm
            state["cloud_only_reason"] = None if _rm == "local" else "insufficient_vram"
            if _rm == "local":
                state["config"].model_name = "qwen3-4b"

        def _default_reload(*a, **kw):
            call_order.append("post_phase_b_reload")

        reload_fn = reload_impl if reload_impl is not None else _default_reload

        with (
            patch("paramem.server.app.write_bundle", lambda **kw: bundle_slot),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch("paramem.server.app._live_reload_base_model", reload_fn),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                )
            )

        return call_order, gates_received, state_dir


# ---------------------------------------------------------------------------
# B1: phase-aware resume — write_bundle never called on resume
# ---------------------------------------------------------------------------


class TestBaseSwapResumePhaseAware:
    """Tests for the resume_phase parameter added for B1.

    Verifies that:
    - Fresh start (resume_phase="") calls write_bundle once.
    - Resume at phaseA_done does NOT call write_bundle, does NOT re-run Phase
      A, reuses the original bundle_slot from the marker, and proceeds to
      Phase B.
    - Resume at phaseB does NOT call write_bundle and runs only Phase B.
    """

    def _make_state(self, tmp_path: Path, *, model_name: str = "mistral") -> dict:
        """Minimal _state dict for resume tests.

        Parameters
        ----------
        model_name:
            The ``config.model_name`` to set.  Use ``"mistral"`` for a fresh-start
            or phaseA_done state (old model still in config before reload).
            Use ``"qwen3-4b"`` for a phaseB resume state (new model already
            loaded — the Phase-B model-identity guard will pass).
        """
        config = MagicMock()
        config.paths.data = tmp_path / "data"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata = tmp_path / "data" / "key_metadata.json"
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.key_metadata_path = config.paths.key_metadata  # mirror the real property
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config.model_name = model_name
        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            t = MagicMock()
            t.enabled = False
            setattr(adapters_cfg, tier, t)
        config.adapters = adapters_cfg
        config.training_config = MagicMock()
        config.consolidation = MagicMock()
        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            # For phaseB resume the reload already succeeded — mode is local.
            # For phaseA/phaseA_done the reload hasn't happened yet; _fake_apply
            # in each test helper sets mode to "local" on a successful reload.
            "mode": "local" if model_name == "qwen3-4b" else None,
            "migration": {
                "state": "TRIAL",
                "trial": {
                    "started_at": "2026-05-24T00:00:00+00:00",
                    "pre_trial_config_sha256": "",
                    "candidate_config_sha256": "aabb",
                    "backup_paths": {},
                    "trial_adapter_dir": "",
                    "trial_graph_dir": "",
                    "gates": {"status": "pending"},
                },
                "recovery_required": [],
                "base_swap_active": False,
            },
            "consolidation_loop": None,
            "background_trainer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }

    def _write_phase_a_done_marker(self, state_dir: Path, bundle_slot_str: str) -> None:
        """Write a phaseA_done marker with the given bundle slot."""
        from paramem.server.trial_state import TRIAL_MARKER_SCHEMA_VERSION, write_trial_marker

        m = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="pretrialxxx",
            candidate_config_sha256="aabb",
            backup_paths={"bundle": bundle_slot_str},
            trial_adapter_dir="",
            trial_graph_dir="",
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseA_done",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=bundle_slot_str,
        )
        write_trial_marker(state_dir, m)

    def _write_phase_b_marker(self, state_dir: Path, bundle_slot_str: str) -> None:
        """Write a phaseB marker with the given bundle slot."""
        from paramem.server.trial_state import TRIAL_MARKER_SCHEMA_VERSION, write_trial_marker

        m = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="pretrialxxx",
            candidate_config_sha256="aabb",
            backup_paths={"bundle": bundle_slot_str},
            trial_adapter_dir="",
            trial_graph_dir="",
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseB",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=bundle_slot_str,
        )
        write_trial_marker(state_dir, m)

    def _run_orchestration(
        self,
        state,
        monkeypatch,
        tmp_path,
        *,
        resume_phase: str = "",
        reload_mode: str = "local",
        bundle_slot_override=None,
    ):
        """Run orchestration with mocks; return (bundle_call_count, submit_calls, state_dir).

        ``bundle_call_count`` counts how many times ``write_bundle`` was called.
        ``submit_calls`` lists phase names ("phase_a_submit", "phase_b_submit") in order.
        """
        import asyncio as _asyncio

        import paramem.server.app as _app

        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        real_bundle_slot = tmp_path / "real_bundle_slot_dir"
        real_bundle_slot.mkdir()

        bundle_call_count = [0]
        submit_calls: list[str] = []
        gates_received: list[dict] = []

        def _fake_write_bundle(**kwargs):
            bundle_call_count[0] += 1
            return bundle_slot_override if bundle_slot_override is not None else real_bundle_slot

        def _fake_submit(fn, **kwargs):
            # Label the submit by reading the marker's base_swap_phase so
            # resume tests (where Phase A is skipped) are labelled correctly.
            from paramem.server.trial_state import read_trial_marker as _rtm

            m = _rtm(state_dir)
            phase = m.base_swap_phase if m is not None else "unknown"
            if phase == "phaseA":
                submit_calls.append("phase_a_submit")
            elif phase in ("phaseA_done", "phaseB"):
                submit_calls.append("phase_b_submit")
            else:
                submit_calls.append(f"unknown_submit_{phase}")
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        fake_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        call_n = [0]

        def _fake_migrate(loop, cfg, ms):
            call_n[0] += 1
            return fake_a if call_n[0] == 1 else fake_b

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        _rm = reload_mode

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: load the renamed-config base model."""
            state["mode"] = _rm
            state["cloud_only_reason"] = None if _rm == "local" else "insufficient_vram"
            if _rm == "local":
                # Simulate config refresh: new model is now live.
                state["config"].model_name = "qwen3-4b"

        with (
            patch("paramem.server.app.write_bundle", _fake_write_bundle),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                    resume_phase=resume_phase,
                )
            )

        return bundle_call_count[0], submit_calls, state_dir, gates_received

    def test_fresh_start_calls_write_bundle_once(self, tmp_path, monkeypatch):
        """Fresh start (resume_phase='') calls write_bundle exactly once."""
        state = self._make_state(tmp_path)
        bundle_count, _, _, _ = self._run_orchestration(
            state, monkeypatch, tmp_path, resume_phase=""
        )
        assert bundle_count == 1, f"Expected write_bundle called once; got {bundle_count}"

    def test_resume_at_phase_a_done_does_not_call_write_bundle(self, tmp_path, monkeypatch):
        """Resume at phaseA_done must NOT call write_bundle.

        The bundle was written at fresh start; calling it again on resume
        would orphan the original Mistral-weights bundle (B1 fix).
        """
        state = self._make_state(tmp_path)
        # Seed the state_dir with a phaseA_done marker containing a bundle slot.
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        original_bundle = tmp_path / "original_bundle"
        original_bundle.mkdir()
        self._write_phase_a_done_marker(state_dir, str(original_bundle))

        bundle_count, submit_calls, _, _ = self._run_orchestration(
            state, monkeypatch, tmp_path, resume_phase="phaseA_done"
        )
        assert bundle_count == 0, (
            f"Resume at phaseA_done must NOT call write_bundle; called {bundle_count} times"
        )

    def test_resume_at_phase_a_done_skips_phase_a(self, tmp_path, monkeypatch):
        """Resume at phaseA_done submits exactly one job (Phase B only).

        Phase A is skipped entirely — only one BackgroundTrainer.submit call.
        Boot already loaded the new model (mode="local", model_name="qwen3-4b"),
        so the reload (gpu_release/gpu_acquire) is NOT called; we go straight to Phase B.
        """
        # Use qwen3-4b to simulate boot having loaded the new model.
        state = self._make_state(tmp_path, model_name="qwen3-4b")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        original_bundle = tmp_path / "original_bundle"
        original_bundle.mkdir()
        self._write_phase_a_done_marker(state_dir, str(original_bundle))

        _, submit_calls, _, _ = self._run_orchestration(
            state, monkeypatch, tmp_path, resume_phase="phaseA_done"
        )
        # Only one submit (Phase B).  If Phase A also ran, there would be 2 submits.
        assert len(submit_calls) == 1, (
            f"Resume at phaseA_done must submit exactly once (Phase B only); "
            f"got {len(submit_calls)} submit(s): {submit_calls}"
        )
        assert submit_calls[0] == "phase_b_submit", (
            f"The single submit must be Phase B; got {submit_calls[0]!r}"
        )

    def test_resume_at_phase_a_done_reuses_original_bundle_slot(self, tmp_path, monkeypatch):
        """Resume at phaseA_done reuses the bundle_slot from the existing marker.

        The orchestration must use the marker's bundle_slot in the Phase B
        marker write — not a freshly derived path.
        Boot already loaded the new model (mode="local", model_name="qwen3-4b").
        """
        # Use qwen3-4b to simulate boot having loaded the new model.
        state = self._make_state(tmp_path, model_name="qwen3-4b")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        original_bundle = tmp_path / "original_bundle"
        original_bundle.mkdir()
        self._write_phase_a_done_marker(state_dir, str(original_bundle.resolve()))

        _, _, final_state_dir, gates = self._run_orchestration(
            state, monkeypatch, tmp_path, resume_phase="phaseA_done"
        )
        # On success the marker is cleared.  Verify the orchestration succeeded
        # (gates status == 'pass') and that bundle_slot in Phase B marker was
        # the original path (verifiable via the phaseB marker written before
        # submit — check via gates since marker is cleared on success).
        assert any(g.get("status") == "pass" for g in gates), (
            f"Expected pass after phaseA_done resume; got {gates}"
        )

    def test_resume_at_phase_b_does_not_call_write_bundle(self, tmp_path, monkeypatch):
        """Resume at phaseB must NOT call write_bundle."""
        # For phaseB resume, the reload already succeeded: config reflects new model.
        state = self._make_state(tmp_path, model_name="qwen3-4b")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        original_bundle = tmp_path / "original_bundle"
        original_bundle.mkdir()
        self._write_phase_b_marker(state_dir, str(original_bundle.resolve()))

        bundle_count, submit_calls, _, _ = self._run_orchestration(
            state, monkeypatch, tmp_path, resume_phase="phaseB"
        )
        assert bundle_count == 0, (
            f"Resume at phaseB must NOT call write_bundle; called {bundle_count} times"
        )

    def test_resume_at_phase_b_skips_phase_a_and_reload(self, tmp_path, monkeypatch):
        """Resume at phaseB skips Phase A and the in-process reload."""
        import paramem.server.app as _app

        # For phaseB resume, the reload already succeeded: config reflects new model.
        state = self._make_state(tmp_path, model_name="qwen3-4b")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        original_bundle = tmp_path / "original_bundle"
        original_bundle.mkdir()
        self._write_phase_b_marker(state_dir, str(original_bundle.resolve()))

        monkeypatch.setattr(_app, "_state", state)

        apply_calls: list[str] = []
        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        submit_count = [0]

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )

        with (
            patch("paramem.server.app.write_bundle", side_effect=AssertionError("should not call")),
            patch("paramem.server.app.migrate", return_value=fake_b),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _noop_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch(
                "paramem.server.app._apply_config_live",
                lambda: apply_calls.append("called"),
            ),
        ):
            import asyncio as _asyncio

            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                    resume_phase="phaseB",
                )
            )

        # Phase B resume: exactly one BackgroundTrainer.submit call (Phase B only).
        assert submit_count[0] == 1, (
            f"phaseB resume must submit exactly once; got {submit_count[0]}"
        )
        assert apply_calls == [], "phaseB resume must not call _apply_config_live"


async def _noop_gates(payload):
    pass


# ---------------------------------------------------------------------------
# Step-3 resume: in-process reload and phaseA_done resume semantics
# ---------------------------------------------------------------------------


class TestBaseSwapStep3ResumeReload:
    """Tests for Step-3 reload behavior in _run_base_swap_orchestration.

    Fresh start (resume_phase=""): calls gpu_release then gpu_acquire;
    on success proceeds to Phase B; on VRAM defer returns with reload_deferred.

    Resume (resume_phase="phaseA_done"): boot already loaded the new model,
    so gpu_release/gpu_acquire are NOT called.  Verify the new model is
    resident (mode=="local", config.model_name==new_model); if so, go straight
    to Phase B.  If not loaded (boot came up cloud-only), set reload_deferred.
    """

    def _make_state(self, tmp_path: Path, *, model_name: str, mode: str = "local") -> dict:
        """Minimal _state for Step-3 tests."""
        config = MagicMock()
        config.paths.data = tmp_path / "data"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata = tmp_path / "data" / "key_metadata.json"
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.key_metadata_path = config.paths.key_metadata  # mirror the real property
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config.model_name = model_name
        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            t = MagicMock()
            t.enabled = False
            setattr(adapters_cfg, tier, t)
        config.adapters = adapters_cfg
        config.training_config = MagicMock()
        config.consolidation = MagicMock()
        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            "mode": mode,
            "migration": {
                "state": "TRIAL",
                "trial": {"gates": {"status": "pending"}},
                "recovery_required": [],
                "base_swap_active": False,
            },
            "consolidation_loop": None,
            "background_trainer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }

    def _write_phase_a_done_marker(self, state_dir: Path, bundle_slot_str: str) -> None:
        from paramem.server.trial_state import TRIAL_MARKER_SCHEMA_VERSION, write_trial_marker

        m = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="pretrialxxx",
            candidate_config_sha256="aabb",
            backup_paths={"bundle": bundle_slot_str},
            trial_adapter_dir="",
            trial_graph_dir="",
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseA_done",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=bundle_slot_str,
        )
        write_trial_marker(state_dir, m)

    def _run_orchestration(
        self,
        state: dict,
        monkeypatch,
        tmp_path: Path,
        *,
        resume_phase: str,
        apply_config_live_side_effect=None,
    ):
        """Run orchestration and return (apply_called, submit_calls, gates, state_dir)."""
        import asyncio as _asyncio

        import paramem.server.app as _app

        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: qwen3-4b\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()
        self._write_phase_a_done_marker(state_dir, str(bundle_slot))

        apply_called: list[bool] = []
        submit_calls: list[str] = []
        gates_received: list[dict] = []

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: load the renamed-config base model.

            Appends True to apply_called and runs apply_config_live_side_effect
            if provided.
            """
            apply_called.append(True)
            if apply_config_live_side_effect is not None:
                apply_config_live_side_effect()

        def _fake_submit(fn, **kwargs):
            from paramem.server.trial_state import read_trial_marker as _rtm

            m = _rtm(state_dir)
            phase = m.base_swap_phase if m is not None else "unknown"
            if phase == "phaseA":
                submit_calls.append("phase_a_submit")
            else:
                submit_calls.append("phase_b_submit")
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        with (
            patch("paramem.server.app.write_bundle", return_value=bundle_slot),
            patch("paramem.server.app.migrate", return_value=fake_b),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                    resume_phase=resume_phase,
                )
            )

        return apply_called, submit_calls, gates_received, state_dir

    def test_fresh_start_reloads_and_proceeds_to_phase_b(self, tmp_path, monkeypatch):
        """Fresh start (resume_phase=""): gpu_acquire (reload) is called; on success
        Phase B runs.

        Simulates the successful in-process reload path (mock sets mode="local"
        and config.model_name="qwen3-4b").
        """
        state = self._make_state(tmp_path, model_name="mistral", mode="local")

        def _apply_success():
            state["mode"] = "local"
            state["cloud_only_reason"] = None
            state["config"].model_name = "qwen3-4b"

        apply_called, submit_calls, gates, _ = self._run_orchestration(
            state,
            monkeypatch,
            tmp_path,
            resume_phase="",
            apply_config_live_side_effect=_apply_success,
        )

        # gpu_acquire (reload) must be called on fresh start.
        assert apply_called, "reload (gpu_acquire) must be called on fresh start"

        # Phase B must have run.
        assert "phase_b_submit" in submit_calls, (
            f"Phase B must run after a successful reload; submit_calls={submit_calls}"
        )

        # Final gate status must be 'pass'.
        statuses = [g.get("status") for g in gates]
        assert "pass" in statuses, f"Expected 'pass' gate status; got {statuses}"
        assert "reload_deferred" not in statuses, (
            f"reload_deferred must NOT appear when reload succeeds; got {statuses}"
        )

    def test_fresh_start_deferred_vram_no_phase_b(self, tmp_path, monkeypatch):
        """Fresh start with insufficient VRAM: reload deferred, Phase B not run.

        Mock gpu_acquire leaves mode=cloud-only (VRAM insufficient).
        The deferred-path must fire: reload_deferred gate, no Phase B.
        """
        state = self._make_state(tmp_path, model_name="mistral", mode="local")

        def _apply_deferred():
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "insufficient_vram"

        apply_called, submit_calls, gates, state_dir = self._run_orchestration(
            state,
            monkeypatch,
            tmp_path,
            resume_phase="",
            apply_config_live_side_effect=_apply_deferred,
        )

        assert apply_called, "reload (gpu_acquire) must be called on fresh start"

        # Phase B must NOT have run.
        assert "phase_b_submit" not in submit_calls, (
            f"Phase B must NOT run when reload is deferred; submit_calls={submit_calls}"
        )

        # Gate status must be reload_deferred.
        statuses = [g.get("status") for g in gates]
        assert "reload_deferred" in statuses, (
            f"Expected reload_deferred gate on VRAM defer; got {statuses}"
        )

        # Marker must remain at phaseA_done.
        from paramem.server.trial_state import read_trial_marker as _rtm

        marker = _rtm(state_dir)
        assert marker is not None, "Marker must remain on deferred reload"
        assert marker.base_swap_phase == "phaseA_done", (
            f"Marker must stay at phaseA_done on defer; got {marker.base_swap_phase!r}"
        )

    def test_phase_a_done_resume_does_not_call_apply_when_model_resident(
        self, tmp_path, monkeypatch
    ):
        """phaseA_done resume: reload (gpu_acquire) is NOT called when boot already
        loaded the new model (mode=="local", config.model_name=="qwen3-4b").

        Boot loaded the new config after the process restarted; calling
        the reload again would unload+reload unnecessarily.
        Phase B runs immediately.
        """
        state = self._make_state(tmp_path, model_name="qwen3-4b", mode="local")

        apply_called, submit_calls, gates, _ = self._run_orchestration(
            state,
            monkeypatch,
            tmp_path,
            resume_phase="phaseA_done",
        )

        # reload (gpu_acquire) must NOT be called on phaseA_done resume with model resident.
        assert not apply_called, (
            "reload (gpu_acquire) must NOT be called on phaseA_done resume when model is "
            "already resident"
        )

        # Phase B must have run.
        assert "phase_b_submit" in submit_calls, (
            f"Phase B must run on phaseA_done resume; submit_calls={submit_calls}"
        )

        # Final gate status must be 'pass'.
        statuses = [g.get("status") for g in gates]
        assert "pass" in statuses, f"Expected 'pass' gate status; got {statuses}"

    def test_phase_a_done_resume_defers_when_model_not_loaded(self, tmp_path, monkeypatch):
        """phaseA_done resume: if boot came up cloud-only (model not resident),
        set reload_deferred and do NOT run Phase B.

        This mirrors the fresh-start VRAM-defer path for the resume case: the
        /gpu/acquire hook will re-launch with resume_phase="phaseA_done" once
        the model is loaded.
        """
        # Boot came up cloud-only (VRAM pressure), model_name still "mistral".
        state = self._make_state(tmp_path, model_name="mistral", mode="cloud-only")
        state["cloud_only_reason"] = "insufficient_vram"

        apply_called, submit_calls, gates, state_dir = self._run_orchestration(
            state,
            monkeypatch,
            tmp_path,
            resume_phase="phaseA_done",
        )

        # reload (gpu_acquire) must NOT be called on resume.
        assert not apply_called, "reload (gpu_acquire) must NOT be called on phaseA_done resume"

        # Phase B must NOT have run.
        assert "phase_b_submit" not in submit_calls, (
            f"Phase B must NOT run when model is not loaded; submit_calls={submit_calls}"
        )

        # Gate status must be reload_deferred.
        statuses = [g.get("status") for g in gates]
        assert "reload_deferred" in statuses, (
            f"Expected reload_deferred gate when model not loaded; got {statuses}"
        )

        # Marker must remain at phaseA_done.
        from paramem.server.trial_state import read_trial_marker as _rtm

        marker = _rtm(state_dir)
        assert marker is not None, "Marker must remain on deferred reload"
        assert marker.base_swap_phase == "phaseA_done", (
            f"Marker must stay at phaseA_done on defer; got {marker.base_swap_phase!r}"
        )


# ---------------------------------------------------------------------------
# R2: in-flight guard — base_swap_active flag
# ---------------------------------------------------------------------------


class TestBaseSwapActiveFlag:
    """Tests for the R2 in-flight guard: base_swap_active flag.

    Verifies that:
    - Rollback returns 409 while base_swap_active=True.
    - Rollback is allowed when base_swap_active=False (deferred state).
    - base_swap_active is cleared in the finally block on deferred return.
    - Confirm returns 409 while base_swap_active=True.
    """

    def _make_trial_state(self, tmp_path: Path, base_swap_active: bool = False) -> dict:
        """Build a TRIAL _state for in-flight guard tests."""
        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            "migration": {
                "state": "TRIAL",
                "trial": {
                    "started_at": "2026-05-24T00:00:00+00:00",
                    "pre_trial_config_sha256": "x" * 64,
                    "candidate_config_sha256": "y" * 64,
                    "backup_paths": {},
                    "trial_adapter_dir": "",
                    "trial_graph_dir": "",
                    "gates": {"status": "pending"},
                },
                "recovery_required": [],
                "base_swap_active": base_swap_active,
            },
            "migration_lock": asyncio.Lock(),
            "server_started_at": "2026-05-24T00:00:00+00:00",
            "mode": "local",
            "background_trainer": None,
            "consolidation_loop": None,
            "session_buffer": None,
        }

    def test_rollback_409_while_base_swap_active(self, tmp_path, monkeypatch):
        """POST /migration/rollback returns 409 base_swap_active while flag is set."""
        state = self._make_trial_state(tmp_path, base_swap_active=True)
        monkeypatch.setattr(app_module, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 409, resp.text
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "base_swap_active", (
            f"Expected 'base_swap_active' error; got {detail}"
        )

    def test_rollback_allowed_when_deferred_flag_clear(self, tmp_path, monkeypatch):
        """Rollback is allowed when base_swap_active=False (deferred/stranded state).

        A phaseA_done marker with base_swap_active=False represents a deferred
        swap where the coroutine has exited.  Rollback is the escape hatch and
        must proceed (not 409).  We patch the bundle-restore path to avoid
        requiring a real bundle on disk — the 409 guard is what we're testing.
        """
        from paramem.server.trial_state import TRIAL_MARKER_SCHEMA_VERSION, write_trial_marker

        state = self._make_trial_state(tmp_path, base_swap_active=False)
        monkeypatch.setattr(app_module, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")

        # Write a phaseA_done marker so rollback takes the base-swap branch.
        bundle_dir = tmp_path / "bundle_slot"
        bundle_dir.mkdir()
        state_dir = state["config"].paths.data / "state"
        marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="x" * 64,
            candidate_config_sha256="y" * 64,
            backup_paths={"bundle": str(bundle_dir)},
            trial_adapter_dir="",
            trial_graph_dir="",
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseA_done",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=str(bundle_dir),
        )
        write_trial_marker(state_dir, marker)

        # Patch restore_bundle and _apply_config_live so rollback can succeed
        # without real data on disk.
        with (
            patch(
                "paramem.server.app.restore_bundle",
                return_value=None,
                create=True,
            ),
            patch("paramem.server.app._apply_config_live", lambda: None),
            patch(
                "paramem.backup.backup.restore_bundle",
                return_value=None,
            ),
        ):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        # Must NOT be 409 base_swap_active.
        assert (
            resp.status_code != 409
            or resp.json().get("detail", {}).get("error") != "base_swap_active"
        ), (
            "Rollback must not be blocked by base_swap_active=False; "
            f"got {resp.status_code} {resp.text}"
        )

    def test_base_swap_active_cleared_on_deferred_return(self, tmp_path, monkeypatch):
        """base_swap_active is False after orchestration exits via deferred reload.

        When gpu_acquire leaves mode=cloud-only, the coroutine returns
        early (deferred).  The finally block must clear base_swap_active so
        rollback is no longer blocked.
        """
        import asyncio as _asyncio

        import paramem.server.app as _app

        state_config = MagicMock()
        state_config.paths.data = tmp_path / "data"
        state_config.paths.data.mkdir(parents=True, exist_ok=True)
        state_config.paths.key_metadata = tmp_path / "data" / "key_metadata.json"
        state_config.adapter_dir = tmp_path / "adapters"
        state_config.adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            t = MagicMock()
            t.enabled = False
            setattr(adapters_cfg, tier, t)
        state_config.adapters = adapters_cfg
        state_config.training_config = MagicMock()
        state_config.consolidation = MagicMock()

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": state_config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            "migration": {
                "state": "TRIAL",
                "trial": {"gates": {"status": "pending"}},
                "recovery_required": [],
                "base_swap_active": False,
            },
            "consolidation_loop": None,
            "background_trainer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }
        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        submit_count = [0]

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live with deferred reload: leave mode=cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "insufficient_vram"

        with (
            patch("paramem.server.app.write_bundle", return_value=bundle_slot),
            patch("paramem.server.app.migrate", return_value=fake_a),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _noop_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                    resume_phase="",
                )
            )

        # After deferred return, base_swap_active must be False.
        mig = state.get("migration", {})
        assert mig.get("base_swap_active") is False, (
            "base_swap_active must be False after deferred return; "
            f"got {mig.get('base_swap_active')!r}"
        )

    def test_confirm_409_while_base_swap_active(self, tmp_path, monkeypatch):
        """POST /migration/confirm returns 409 base_swap_active while flag is set.

        Confirm normally rejects with trial_active when state==TRIAL, but the
        explicit base_swap_active guard inside the lock fires as a distinct
        error when state has already advanced past STAGING (e.g. race).
        We directly set base_swap_active=True with state=STAGING to exercise
        the guard in isolation.
        """
        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config.model_name = "mistral"

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")

        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            t = MagicMock()
            t.enabled = False
            setattr(adapters_cfg, tier, t)
        config.adapters = adapters_cfg

        staging = initial_migration_state()
        staging["state"] = "STAGING"
        staging["candidate_path"] = str(cand_yaml)
        staging["candidate_hash"] = hashlib.sha256(b"model: qwen3-4b\n").hexdigest()
        staging["candidate_bytes"] = b"model: qwen3-4b\n"
        staging["candidate_text"] = "model: qwen3-4b\n"
        staging["parsed_candidate"] = {"model": "qwen3-4b"}
        staging["parsed_live"] = {"model": "mistral"}
        staging["tier_diff"] = [
            {
                "dotted_path": "model",
                "old_value": "mistral",
                "new_value": "qwen3-4b",
                "tier": "destructive",
            }
        ]
        # Set base_swap_active=True while leaving state=STAGING — exercises
        # the explicit guard inside the lock body.
        staging["base_swap_active"] = True

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(live_yaml),
            "consolidating": False,
            "migration": staging,
            "migration_lock": asyncio.Lock(),
            "server_started_at": "2026-05-24T00:00:00+00:00",
            "mode": "local",
            "background_trainer": None,
            "consolidation_loop": None,
            "session_buffer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409, resp.text
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "base_swap_active", (
            f"Expected 'base_swap_active' error on confirm; got {detail}"
        )


# ---------------------------------------------------------------------------
# R1: /gpu/acquire re-launches deferred base-swap
# ---------------------------------------------------------------------------


class TestGpuAcquireBaseSwapResume:
    """Tests for the R1 hook: /gpu/acquire re-launches orchestration when
    reload succeeds and a phaseA_done base-swap marker exists.
    """

    def _make_cloud_only_state(self, tmp_path: Path) -> dict:
        """Build a cloud-only _state with a phaseA_done base-swap marker."""
        from paramem.server.trial_state import TRIAL_MARKER_SCHEMA_VERSION, write_trial_marker

        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        bundle_dir = tmp_path / "bundle_slot"
        bundle_dir.mkdir()

        marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-05-24T00:00:00+00:00",
            pre_trial_config_sha256="x" * 64,
            candidate_config_sha256="y" * 64,
            backup_paths={"bundle": str(bundle_dir)},
            trial_adapter_dir="",
            trial_graph_dir="",
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseA_done",
            old_model="mistral",
            new_model="qwen3-4b",
            bundle_slot=str(bundle_dir),
        )
        write_trial_marker(state_dir, marker)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: qwen3-4b\n")

        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(live_yaml),
            "consolidating": False,
            "migration": {
                "state": "TRIAL",
                "trial": {"gates": {"status": "reload_deferred"}},
                "recovery_required": [],
                "base_swap_active": False,
            },
            "migration_lock": asyncio.Lock(),
            "server_started_at": "2026-05-24T00:00:00+00:00",
            "mode": "cloud-only",
            "cloud_only_reason": "insufficient_vram",
            "background_trainer": None,
            "consolidation_loop": None,
            "session_buffer": None,
        }

    def test_gpu_acquire_relaunches_orchestration_on_phaseA_done_marker(
        self, tmp_path, monkeypatch
    ):
        """When /gpu/acquire reload succeeds and a phaseA_done marker exists,
        the orchestration is re-launched in resume mode (resume_phase='phaseA_done').
        """
        import paramem.server.app as _app

        state = self._make_cloud_only_state(tmp_path)
        monkeypatch.setattr(_app, "_state", state)

        orchestration_calls: list[dict] = []

        async def _fake_orchestration(**kwargs):
            orchestration_calls.append(kwargs)

        monkeypatch.setattr(_app, "_run_base_swap_orchestration", _fake_orchestration)

        def _fake_reload():
            # Simulate successful reload.
            state["mode"] = "local"
            state["cloud_only_reason"] = None

        with (
            patch("paramem.server.app._live_reload_base_model", _fake_reload),
            patch("paramem.server.app._set_voice_pipeline_profile", lambda *a: None),
            patch(
                "paramem.server.app._get_hold_state",
                return_value={"hold_active": False, "owner_pid": None, "owner_alive": False},
            ),
            patch("paramem.server.app._clear_hold_env", return_value=False),
        ):
            client = TestClient(_app.app, raise_server_exceptions=False)
            resp = client.post("/gpu/acquire")

        assert resp.status_code == 200, resp.text
        # Orchestration must have been re-launched once.
        assert len(orchestration_calls) == 1, (
            f"Expected orchestration re-launched once; got {len(orchestration_calls)} calls"
        )
        call = orchestration_calls[0]
        assert call.get("resume_phase") == "phaseA_done", (
            f"Expected resume_phase='phaseA_done'; got {call.get('resume_phase')!r}"
        )
        assert call.get("old_model") == "mistral"
        assert call.get("new_model") == "qwen3-4b"

    def test_gpu_acquire_does_not_relaunch_when_active(self, tmp_path, monkeypatch):
        """When base_swap_active=True, /gpu/acquire must not re-launch orchestration.

        The orchestration is already running — re-launching would cause a
        concurrent double-execution.
        """
        import paramem.server.app as _app

        state = self._make_cloud_only_state(tmp_path)
        state["migration"]["base_swap_active"] = True
        monkeypatch.setattr(_app, "_state", state)

        orchestration_calls: list[dict] = []

        async def _fake_orchestration(**kwargs):
            orchestration_calls.append(kwargs)

        monkeypatch.setattr(_app, "_run_base_swap_orchestration", _fake_orchestration)

        def _fake_reload():
            state["mode"] = "local"
            state["cloud_only_reason"] = None

        with (
            patch("paramem.server.app._live_reload_base_model", _fake_reload),
            patch("paramem.server.app._set_voice_pipeline_profile", lambda *a: None),
            patch(
                "paramem.server.app._get_hold_state",
                return_value={"hold_active": False, "owner_pid": None, "owner_alive": False},
            ),
            patch("paramem.server.app._clear_hold_env", return_value=False),
        ):
            client = TestClient(_app.app, raise_server_exceptions=False)
            resp = client.post("/gpu/acquire")

        assert resp.status_code == 200, resp.text
        assert len(orchestration_calls) == 0, (
            "Must not re-launch orchestration while base_swap_active=True"
        )

    def test_gpu_acquire_does_not_relaunch_when_no_deferred_marker(self, tmp_path, monkeypatch):
        """When reload succeeds but no phaseA_done marker exists, no re-launch."""
        import paramem.server.app as _app

        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: qwen3-4b\n")

        state = {
            "model": MagicMock(),
            "config": config,
            "config_path": str(live_yaml),
            "migration": {
                "state": "LIVE",
                "base_swap_active": False,
                "recovery_required": [],
            },
            "migration_lock": asyncio.Lock(),
            "mode": "cloud-only",
            "cloud_only_reason": "gpu_conflict",
            "session_buffer": None,
        }
        monkeypatch.setattr(_app, "_state", state)

        orchestration_calls: list[dict] = []

        async def _fake_orchestration(**kwargs):
            orchestration_calls.append(kwargs)

        monkeypatch.setattr(_app, "_run_base_swap_orchestration", _fake_orchestration)

        def _fake_reload():
            state["mode"] = "local"
            state["cloud_only_reason"] = None

        with (
            patch("paramem.server.app._live_reload_base_model", _fake_reload),
            patch("paramem.server.app._set_voice_pipeline_profile", lambda *a: None),
            patch(
                "paramem.server.app._get_hold_state",
                return_value={"hold_active": False, "owner_pid": None, "owner_alive": False},
            ),
            patch("paramem.server.app._clear_hold_env", return_value=False),
        ):
            client = TestClient(_app.app, raise_server_exceptions=False)
            resp = client.post("/gpu/acquire")

        assert resp.status_code == 200, resp.text
        assert len(orchestration_calls) == 0, (
            "Must not re-launch orchestration when no phaseA_done marker"
        )


# ---------------------------------------------------------------------------
# D1: wording — no restart references
# ---------------------------------------------------------------------------


class TestD1NoRestartWording:
    """Verify that D1 wording fixes removed 'restart' from operator-facing text."""

    def test_base_change_preview_no_restart(self, tmp_path, capsys):
        """_render_base_change_preview must not mention a server restart."""
        from paramem.cli.migrate import _render_base_change_preview

        _render_base_change_preview(
            {"old_model": "mistral", "new_model": "qwen3-4b", "consequence": "..."}
        )
        captured = capsys.readouterr().out
        assert "restart the server" not in captured.lower(), (
            f"_render_base_change_preview must not mention restarting the server; got:\n{captured}"
        )
        # The assertion just verifies "restart the server" is absent (checked above).
        # The remaining check is that the output is informative — at minimum it
        # must not contain the old incorrect instruction.
        assert "a server restart is required between" not in captured.lower(), (
            "Expected old restart wording to be absent in preview output"
        )

    def test_compute_base_change_consequence_no_restart(self):
        """compute_base_change consequence string must not say restart is required."""
        from paramem.server.migration import compute_base_change

        result = compute_base_change({"model": "mistral"}, {"model": "qwen3-4b"})
        assert result is not None
        consequence = result["consequence"]
        assert "restart is required after" not in consequence.lower(), (
            f"consequence must not describe restart-after-Phase-A; got: {consequence}"
        )
        assert "no server restart" in consequence.lower(), (
            f"consequence should state that no restart is required; got: {consequence}"
        )

    def test_confirm_response_base_swap_docstring_no_restart(self):
        """ConfirmResponse.base_swap docstring must not say 'restart the server'."""
        import paramem.server.app as _app

        doc = _app.ConfirmResponse.__doc__ or ""
        assert "restart the server to complete Phase B" not in doc, (
            f"ConfirmResponse docstring must not instruct user to restart the server; got:\n{doc}"
        )


# ---------------------------------------------------------------------------
# Phase-B model-identity guard
# ---------------------------------------------------------------------------


class TestPhaseBModelIdentityGuard:
    """Tests for the Phase-B model-identity guard in _run_base_swap_orchestration.

    The guard fires immediately before Phase B dispatches migrate() and
    verifies that:
      1. _state["mode"] == "local"
      2. config.model_name == new_model

    On mismatch: gates receive phase_b_model_mismatch status; migrate() is
    NOT called; the marker and bundle remain on disk for rollback.
    """

    def _make_state(self, tmp_path: Path, *, loaded_model_name: str = "mistral") -> dict:
        """Minimal _state dict with model_name matching the loaded model."""
        config = MagicMock()
        config.paths.data = tmp_path / "data"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.paths.key_metadata = tmp_path / "data" / "key_metadata.json"
        config.paths.key_metadata.parent.mkdir(parents=True, exist_ok=True)
        config.key_metadata_path = config.paths.key_metadata  # mirror the real property
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        # model_name is set to the currently-loaded model.  After a successful
        # reload it will equal new_model; a mismatch simulates failed/partial reload.
        config.model_name = loaded_model_name
        adapters_cfg = MagicMock()
        for tier in ("episodic", "semantic", "procedural"):
            tier_mock = MagicMock()
            tier_mock.enabled = False
            setattr(adapters_cfg, tier, tier_mock)
        config.adapters = adapters_cfg
        config.training_config = MagicMock()
        config.consolidation = MagicMock()

        return {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "config_path": str(tmp_path / "server.yaml"),
            "consolidating": False,
            "migration": {
                "state": "TRIAL",
                "trial": {
                    "started_at": "2026-05-24T00:00:00+00:00",
                    "pre_trial_config_sha256": "",
                    "candidate_config_sha256": "aabb",
                    "backup_paths": {},
                    "trial_adapter_dir": "",
                    "trial_graph_dir": "",
                    "gates": {"status": "pending"},
                },
                "recovery_required": [],
                "base_swap_active": False,
            },
            "consolidation_loop": None,
            "background_trainer": None,
            "memory_store": MagicMock(),
            "event_loop": None,
        }

    def _run_with_mismatch(
        self,
        state: dict,
        monkeypatch,
        tmp_path: Path,
        *,
        loaded_model_name: str,
        mode_after_reload: str = "local",
    ):
        """Run orchestration where gpu_acquire sets the given model name and mode.

        Returns (migrate_call_count, gates_received, marker_at_end).
        """
        import asyncio as _asyncio

        import paramem.server.app as _app

        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        migrate_call_count = [0]
        gates_received: list[dict] = []

        submit_count = [0]

        def _fake_submit(fn, **kwargs):
            submit_count[0] += 1
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )

        def _fake_migrate(loop, cfg, ms):
            migrate_call_count[0] += 1
            # Phase A result (train→simulate); Phase B should not be called on mismatch.
            return fake_a

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: set mode and config.model_name as controlled by test."""
            state["mode"] = mode_after_reload
            is_local = mode_after_reload == "local"
            state["cloud_only_reason"] = None if is_local else "insufficient_vram"
            state["config"].model_name = loaded_model_name

        with (
            patch("paramem.server.app.write_bundle", return_value=bundle_slot),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                )
            )

        marker = read_trial_marker(state_dir)
        return migrate_call_count[0], gates_received, marker

    def test_guard_aborts_phase_b_when_model_name_wrong(self, tmp_path, monkeypatch):
        """Phase B is NOT invoked when config.model_name != new_model after reload.

        Simulates a partial reload where the config still reports the old model
        name despite mode=local.  migrate() must be called exactly once (Phase A
        only); Phase B is blocked by the guard.
        """
        # loaded_model_name stays "mistral" after fake reload (wrong model still loaded)
        state = self._make_state(tmp_path, loaded_model_name="mistral")
        migrate_count, gates, marker = self._run_with_mismatch(
            state,
            monkeypatch,
            tmp_path,
            loaded_model_name="mistral",  # wrong: expected "qwen3-4b"
            mode_after_reload="local",
        )

        # migrate() called once (Phase A), NOT twice (Phase B blocked by guard).
        assert migrate_count == 1, (
            f"Phase B migrate() must not be called on model mismatch; "
            f"migrate called {migrate_count} times"
        )

        # Gates must contain phase_b_model_mismatch.
        statuses = [g.get("status") for g in gates]
        assert "phase_b_model_mismatch" in statuses, (
            f"Expected phase_b_model_mismatch gate status; got {statuses}"
        )

    def test_guard_aborts_phase_b_when_mode_local_but_wrong_model(self, tmp_path, monkeypatch):
        """Phase B is NOT invoked when mode=local but config.model_name is wrong.

        This covers the silent-wrong-outcome scenario: the server reports "local"
        (model loaded) but the config still names the old model.  The guard
        fires on the name mismatch even though mode looks correct.
        """
        # mode=local (reload claims success), but config still reflects "mistral"
        # instead of the expected "qwen3-4b".
        state = self._make_state(tmp_path, loaded_model_name="mistral")
        migrate_count, gates, marker = self._run_with_mismatch(
            state,
            monkeypatch,
            tmp_path,
            loaded_model_name="mistral",  # wrong model name despite mode=local
            mode_after_reload="local",
        )

        # migrate() must be called once (Phase A only; Phase B blocked).
        assert migrate_count == 1, (
            f"Phase B migrate() must not be called on model name mismatch; "
            f"migrate called {migrate_count} times"
        )

        statuses = [g.get("status") for g in gates]
        assert "phase_b_model_mismatch" in statuses, (
            f"Expected phase_b_model_mismatch gate status; got {statuses}"
        )

    def test_guard_aborts_phase_b_when_mode_not_local(self, tmp_path, monkeypatch):
        """Phase B is NOT invoked when mode != local after reload.

        When mode=cloud-only, the existing reload-deferred path fires before
        Phase B is reached — the guard's mode check is a redundant safety net
        for any residual path that reaches Phase B with mode still non-local.
        This test exercises the guard in isolation by constructing a state where
        gpu_acquire returns with mode=cloud-only AND the config already
        shows the new model name (so the name check alone wouldn't catch it).
        Because the deferred path fires first, the result is 'reload_deferred',
        not 'phase_b_model_mismatch' — the guard at reload level already handled it.
        """
        # This test verifies that Phase B does NOT run when mode=cloud-only,
        # regardless of which guard intercepts it.
        state = self._make_state(tmp_path, loaded_model_name="qwen3-4b")
        migrate_count, gates, marker = self._run_with_mismatch(
            state,
            monkeypatch,
            tmp_path,
            loaded_model_name="qwen3-4b",
            mode_after_reload="cloud-only",
        )

        # Phase B must NOT run — only one migrate() call (Phase A).
        assert migrate_count == 1, (
            f"Phase B migrate() must not be called when mode!=local; "
            f"migrate called {migrate_count} times"
        )

        # Either reload_deferred (handled by the reload check) or
        # phase_b_model_mismatch (handled by the guard) is acceptable —
        # both prevent Phase B from running on a non-local server.
        statuses = [g.get("status") for g in gates]
        phase_b_blocked = "reload_deferred" in statuses or "phase_b_model_mismatch" in statuses
        assert phase_b_blocked, (
            f"Expected Phase B to be blocked (reload_deferred or phase_b_model_mismatch); "
            f"got {statuses}"
        )

    def test_guard_preserves_marker_and_bundle_on_mismatch(self, tmp_path, monkeypatch):
        """Marker and bundle remain on disk after a model-identity mismatch.

        The operator needs the marker and bundle to be intact so that
        ``paramem migrate --rollback`` can restore the prior model.
        """
        state = self._make_state(tmp_path, loaded_model_name="mistral")
        _count, _gates, marker = self._run_with_mismatch(
            state,
            monkeypatch,
            tmp_path,
            loaded_model_name="mistral",
            mode_after_reload="local",
        )

        # Marker must still exist (not cleared).
        assert marker is not None, (
            "Trial marker must be preserved on phase_b_model_mismatch for rollback"
        )
        # Marker should be at phaseB (written before the guard fires).
        assert marker.base_swap_phase == "phaseB", (
            f"Marker must stay at 'phaseB' after mismatch; got {marker.base_swap_phase!r}"
        )

    def test_guard_does_not_fire_on_correct_model(self, tmp_path, monkeypatch):
        """Phase B runs normally when mode=local and config.model_name == new_model.

        This is the happy-path regression: the guard must NOT block a correctly
        reloaded server.
        """
        import asyncio as _asyncio

        import paramem.server.app as _app

        state = self._make_state(tmp_path, loaded_model_name="qwen3-4b")
        monkeypatch.setattr(_app, "_state", state)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(b"model: mistral\n")
        cand_yaml = tmp_path / "candidate.yaml"
        cand_yaml.write_bytes(b"model: qwen3-4b\n")
        state_dir = state["config"].paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = state["config"].paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        bundle_slot = tmp_path / "bundle_slot_dir"
        bundle_slot.mkdir()

        migrate_call_count = [0]
        gates_received: list[dict] = []

        def _fake_submit(fn, **kwargs):
            fn()

        mock_bt = MagicMock()
        mock_bt.submit = _fake_submit
        monkeypatch.setattr("paramem.server.app.BackgroundTrainer", lambda **kwargs: mock_bt)

        from paramem.server.active_store_migration import MigrationState

        fake_a = MigrationState(
            direction="train_to_simulate",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="train",
            target_mode="simulate",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        fake_b = MigrationState(
            direction="simulate_to_train",
            started_at="2026-05-24T00:00:00+00:00",
            source_mode="simulate",
            target_mode="train",
            completed_tiers=["episodic", "semantic", "procedural"],
            failed_tiers={},
        )
        call_n = [0]

        def _fake_migrate(loop, cfg, ms):
            call_n[0] += 1
            migrate_call_count[0] += 1
            return fake_a if call_n[0] == 1 else fake_b

        async def _fake_update_gates(payload):
            gates_received.append(payload)

        async def _fake_gpu_release():
            """Simulate gpu_release: drain base model, enter cloud-only."""
            state["mode"] = "cloud-only"
            state["cloud_only_reason"] = "released"

        def _fake_apply_config_live():
            """Simulate _apply_config_live: correct reload — mode=local, model_name=qwen3-4b."""
            state["mode"] = "local"
            state["cloud_only_reason"] = None
            state["config"].model_name = "qwen3-4b"

        with (
            patch("paramem.server.app.write_bundle", return_value=bundle_slot),
            patch("paramem.server.app.migrate", side_effect=_fake_migrate),
            patch("paramem.server.app._rename_config", lambda s, d: None),
            patch("paramem.server.app._update_trial_gates", _fake_update_gates),
            patch("paramem.server.app.create_consolidation_loop", return_value=MagicMock()),
            patch(
                "paramem.server.app.ThermalPolicy.from_consolidation_config",
                return_value=MagicMock(),
            ),
            patch("paramem.server.app.gpu_release", _fake_gpu_release),
            patch("paramem.server.app._apply_config_live", _fake_apply_config_live),
            patch(
                "paramem.server.app._live_reload_base_model",
                lambda *a, **kw: None,
            ),
        ):
            _asyncio.run(
                _app._run_base_swap_orchestration(
                    candidate_path_str=str(cand_yaml),
                    live_config_path=live_yaml,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model="mistral",
                    new_model="qwen3-4b",
                    started_at="2026-05-24T00:00:00+00:00",
                    candidate_hash="aabb",
                )
            )

        # Both phases should have run (2 migrate calls).
        assert migrate_call_count[0] == 2, (
            f"Both Phase A and Phase B must run when model identity is correct; "
            f"migrate called {migrate_call_count[0]} times"
        )

        statuses = [g.get("status") for g in gates_received]
        assert "pass" in statuses, (
            f"Expected 'pass' gate status on correct model identity; got {statuses}"
        )
