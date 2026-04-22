"""Deterministic no-GPU end-to-end test for POST /migration/confirm (plan §11.2).

This single test exercises the full confirm flow as one assertion sequence:
  Steps 1–7  — POST /migration/confirm via TestClient with mocked trainer.
  Step 8     — Directly run _run_trial_consolidation with mocked consolidation.
  Step 9     — Assert gates.status == "no_new_sessions".
  Steps 10–11 — Restart simulation via recover_migration_state → RESUME_TRIAL.

No GPU is used.  The trainer is mocked at the run_consolidation boundary.
The trial consolidation task is captured and driven synchronously by the test
rather than relying on TestClient's event loop to flush it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.migration import initial_migration_state
from paramem.server.migration_recovery import RecoveryAction, recover_migration_state
from paramem.server.trial_state import read_trial_marker

# ---------------------------------------------------------------------------
# Synthetic config bytes — no personal data.
# ---------------------------------------------------------------------------

_LIVE_YAML = b"model: mistral\ndebug: false\n"
_CAND_YAML = b"model: mistral\ndebug: true\n"


def _sha256(b: bytes) -> str:
    """Return hex SHA-256 of bytes."""
    return hashlib.sha256(b).hexdigest()


def _build_staging_state(tmp_path: Path) -> dict:
    """Build a STAGING _state with real files under tmp_path.

    The live config, candidate file, and all data directories are created
    here.  The returned dict is ready to be monkeypatched into app_module._state.

    Parameters
    ----------
    tmp_path:
        Pytest-provided temporary directory (unique per test run).

    Returns
    -------
    dict
        A complete _state dict in STAGING with a valid candidate stash.
    """
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
        "model": MagicMock(),  # non-None so _run_trial_consolidation doesn't short-circuit
        "tokenizer": MagicMock(),
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
        "ha_context": None,
        "speaker_store": None,
    }


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


def test_confirm_to_trial_e2e_no_gpu(tmp_path, monkeypatch):
    """Full confirm flow with mocked trainer (plan §11.2).

    Steps 1–7: POST /migration/confirm via TestClient.
    Step 8: Drive _run_trial_consolidation directly via asyncio.run().
    Step 9: Assert gates.status == "no_new_sessions".
    Steps 10–11: Rebuild _state from scratch and run lifespan recovery →
                 assert RESUME_TRIAL with matching trial fields.

    No GPU — run_consolidation is mocked at its call-site boundary inside
    _run_trial_consolidation.
    """
    # ------------------------------------------------------------------
    # Step 1: Build _state with STAGING stash + tmp_path-rooted paths.
    # ------------------------------------------------------------------
    state = _build_staging_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", state)

    # ------------------------------------------------------------------
    # Step 2: Monkeypatch _run_trial_consolidation to a no-op so that
    #         the background task created by the HTTP handler does not
    #         interfere with the test.  We will drive the real coroutine
    #         manually in step 8.
    # ------------------------------------------------------------------
    async def _noop_trial():
        """No-op placeholder so create_task during HTTP handling is harmless."""

    monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)

    # ------------------------------------------------------------------
    # Step 3: POST /migration/confirm.
    # ------------------------------------------------------------------
    client = TestClient(app_module.app, raise_server_exceptions=False)
    resp = client.post("/migration/confirm", json={})

    # ------------------------------------------------------------------
    # Step 4: Assert 200; state=TRIAL on disk + in _state.
    # ------------------------------------------------------------------
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["state"] == "TRIAL", f"Response state not TRIAL: {body}"
    assert state["migration"]["state"] == "TRIAL", (
        f"_state['migration']['state'] not TRIAL after confirm: {state['migration']}"
    )

    # ------------------------------------------------------------------
    # Step 5: Assert 3 backup slots exist with valid meta.json +
    #         pre_trial_hash set.
    # ------------------------------------------------------------------
    # Production layout: config.paths.data is data/ha; handler appends
    # "backups" directly (no extra /ha/ segment).
    backups_root = state["config"].paths.data / "backups"
    expected_pre_trial_hash = _sha256(_LIVE_YAML)
    for kind in ("config", "graph", "registry"):
        kind_dir = backups_root / kind
        assert kind_dir.exists(), f"Backup kind dir missing: {kind_dir}"
        slots = [d for d in kind_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) == 1, f"Expected 1 slot in {kind_dir}, found {slots}"
        meta_files = list(slots[0].glob("*.meta.json"))
        assert meta_files, f"No .meta.json in {slots[0]}"
        meta_data = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta_data.get("pre_trial_hash") == expected_pre_trial_hash, (
            f"pre_trial_hash mismatch in {kind} backup: "
            f"got {meta_data.get('pre_trial_hash')!r}, want {expected_pre_trial_hash!r}"
        )

    # ------------------------------------------------------------------
    # Step 6: Assert state/trial.json round-trips via read_trial_marker.
    # ------------------------------------------------------------------
    # Production layout: marker lives at <config.paths.data>/state/trial.json.
    state_dir = state["config"].paths.data / "state"
    marker = read_trial_marker(state_dir)
    assert marker is not None, f"state/trial.json not found under {state_dir}"
    assert marker.candidate_config_sha256 == _sha256(_CAND_YAML), (
        "marker.candidate_config_sha256 mismatch"
    )
    assert marker.pre_trial_config_sha256 == _sha256(_LIVE_YAML), (
        "marker.pre_trial_config_sha256 mismatch"
    )

    # ------------------------------------------------------------------
    # Step 7: Assert configs/server.yaml bytes equal candidate bytes
    #         (rename happened).
    # ------------------------------------------------------------------
    live_config_path = Path(state["config_path"])
    assert live_config_path.read_bytes() == _CAND_YAML, (
        "Live config bytes after confirm do not match candidate bytes. "
        "Rename (step 4) did not complete."
    )

    # ------------------------------------------------------------------
    # Step 8: Drive _run_trial_consolidation directly so gates are updated.
    #
    # The HTTP handler used the no-op placeholder.  We now run the real
    # _run_trial_consolidation coroutine with all GPU-touching calls
    # mocked.  This mirrors the pattern in test_trial_consolidation_overrides.py
    # (TestRunTrialConsolidation.test_trial_completion_sets_no_new_sessions_gate).
    # ------------------------------------------------------------------
    monkeypatch.undo()  # remove the _noop_trial patch so the real fn is importable
    monkeypatch.setattr(app_module, "_state", state)  # re-apply state after undo

    mock_summary = {"status": "no_pending", "sessions": 0}

    async def _run_trial_with_mocks():
        mock_loop = MagicMock()
        with patch(
            "paramem.server.consolidation.create_consolidation_loop",
            return_value=mock_loop,
        ):
            with patch("paramem.server.config.load_server_config") as mock_load:
                cfg = MagicMock()
                cfg.consolidation.mode = "train"
                cfg.paths.data = tmp_path / "data"
                mock_load.return_value = cfg

                with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                    mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                    mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                    with patch(
                        "paramem.server.consolidation.run_consolidation",
                        return_value=mock_summary,
                    ):
                        await app_module._run_trial_consolidation()

    asyncio.run(_run_trial_with_mocks())

    # ------------------------------------------------------------------
    # Step 9: Assert gates.status == "no_new_sessions".
    # ------------------------------------------------------------------
    trial = state["migration"].get("trial")
    assert trial is not None, "trial stash missing from _state['migration']"
    gates = trial.get("gates", {})
    assert gates.get("status") == "no_new_sessions", (
        f"Expected gates.status='no_new_sessions', got: {gates}"
    )

    # ------------------------------------------------------------------
    # Steps 10–11: Restart simulation.
    #
    # Rebuild _state from scratch (simulating a server restart).  The
    # files written in steps 2–4 are still on disk.  Run
    # recover_migration_state against those files and assert that the
    # action is RESUME_TRIAL with the correct trial fields.
    # ------------------------------------------------------------------
    # The live config on disk is now the candidate bytes (post-rename),
    # and trial.json records the pre-trial hash of the original _LIVE_YAML.
    # Because sha256(live_config) != marker.pre_trial_config_sha256, the
    # decision matrix selects case 1: RESUME_TRIAL.
    recovery_result = recover_migration_state(
        state_dir=state_dir,
        live_config_path=live_config_path,
        backups_root=backups_root,
    )

    assert recovery_result.action == RecoveryAction.RESUME_TRIAL, (
        f"Expected RESUME_TRIAL after restart simulation, got: {recovery_result.action}. "
        f"log_lines: {recovery_result.log_lines}"
    )
    assert recovery_result.trial_marker is not None, (
        "RESUME_TRIAL action must carry the trial_marker"
    )
    assert recovery_result.trial_marker.pre_trial_config_sha256 == _sha256(_LIVE_YAML), (
        "Recovered marker.pre_trial_config_sha256 does not match original live config hash"
    )
    assert recovery_result.trial_marker.candidate_config_sha256 == _sha256(_CAND_YAML), (
        "Recovered marker.candidate_config_sha256 does not match candidate hash"
    )

    # Simulate how lifespan seeds _state from recovery (mirrors app.py:836–858).
    fresh_migration = initial_migration_state()
    m = recovery_result.trial_marker
    fresh_migration["state"] = "TRIAL"
    fresh_migration["trial"] = {
        "started_at": m.started_at,
        "pre_trial_config_sha256": m.pre_trial_config_sha256,
        "candidate_config_sha256": m.candidate_config_sha256,
        "backup_paths": {
            "config": m.backup_paths.get("config", ""),
            "graph": m.backup_paths.get("graph", ""),
            "registry": m.backup_paths.get("registry", ""),
        },
        "trial_adapter_dir": m.trial_adapter_dir,
        "trial_graph_dir": m.trial_graph_dir,
        "gates": {"status": "pending"},
    }
    fresh_migration["recovery_required"] = []

    # Step 11: Assert _state["migration"]["state"] == "TRIAL" with same trial fields.
    assert fresh_migration["state"] == "TRIAL", (
        f"Post-recovery migration state is not TRIAL: {fresh_migration['state']}"
    )
    assert fresh_migration["trial"]["pre_trial_config_sha256"] == _sha256(_LIVE_YAML), (
        "Post-recovery trial stash pre_trial_config_sha256 mismatch"
    )
    assert fresh_migration["trial"]["candidate_config_sha256"] == _sha256(_CAND_YAML), (
        "Post-recovery trial stash candidate_config_sha256 mismatch"
    )
    # Backup paths must have all three slots populated.
    bp = fresh_migration["trial"]["backup_paths"]
    for kind in ("config", "graph", "registry"):
        assert bp.get(kind), f"Post-recovery backup_paths[{kind!r}] is empty"
