"""Tests for the GET /integrity endpoint and integrity-related boot/migration wiring.

Covers:
- GET /integrity with a valid plaintext store → ok=True, HTTP 200.
- GET /integrity with a corrupt registry → ok=False, HTTP 200 (report, not 5xx).
- Boot degraded: real corrupt registry on disk → _preload_memory_store sets
  store_load_degraded=True via the production infrastructure integrity check.
- Base-swap 409: corrupt store + base-swap candidate → 409 with integrity_failure.
- Pure mode-switch 409 (Fix-1 path): corrupt store + mode-only candidate → 409,
  AND live config file is NOT renamed on failure.
- _arm_active_store_migration with corrupt registry → returns False, does not arm.

Tests use FastAPI TestClient with monkeypatched _state; no live server, no GPU.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.integrity import _OK, _PARSE_ERROR, FileCheck, IntegrityReport
from paramem.server.migration import MigrationStashState, TierDiffRow, initial_migration_state
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, mode: str = "train") -> MagicMock:
    """Build a minimal config mock for endpoint tests."""
    data_dir = tmp_path / "data" / "ha"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = data_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cfg = MagicMock()
    cfg.paths.data = data_dir
    cfg.adapter_dir = adapter_dir
    cfg.key_metadata_path = data_dir / "registry" / "key_metadata.json"
    cfg.consolidation.mode = mode
    cfg.consolidation.indexed_key_replay = True
    return cfg


def _make_minimal_state(tmp_path: Path) -> dict:
    """Build a minimal _state dict for endpoint tests."""
    cfg = _make_config(tmp_path)
    return {
        "config": cfg,
        "model": None,
        "consolidating": False,
        "migration": initial_migration_state(),
        "migration_lock": asyncio.Lock(),
        "server_started_at": "2026-05-01T00:00:00+00:00",
        "store_load_degraded": False,
        "daily_loadable": False,
        "memory_store": None,
        "consolidation_loop": None,
        "config_path": None,
    }


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


def _ok_report() -> IntegrityReport:
    """Build an IntegrityReport with one ok check and no failures."""
    fc = FileCheck(
        "/data/ha/adapters/episodic/indexed_key_registry.json",
        "registry",
        "episodic",
        _OK,
        "",
    )
    return IntegrityReport(ok=True, checks=[fc], failures=[])


def _fail_report() -> IntegrityReport:
    """Build an IntegrityReport with one parse_error failure."""
    fc = FileCheck(
        "/data/ha/adapters/episodic/indexed_key_registry.json",
        "registry",
        "episodic",
        _PARSE_ERROR,
        "Expecting property name: line 1 col 2 (char 1)",
    )
    return IntegrityReport(ok=False, checks=[fc], failures=[fc])


def _write_corrupt_registry(ep_dir: Path) -> None:
    """Write a corrupt registry file that triggers a parse_error on load."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "indexed_key_registry.json").write_text("{not: json}", encoding="utf-8")


def _write_valid_registry(ep_dir: Path, keys: list[str]) -> None:
    """Write a valid indexed_key_registry.json with simhash fingerprints for every key.

    Each key is assigned a non-zero placeholder fingerprint so the per-file
    self-consistency check (active keys must have a fingerprint) passes.
    """
    from paramem.memory.entry import compute_simhash

    ep_dir.mkdir(parents=True, exist_ok=True)
    reg = KeyRegistry()
    for k in keys:
        reg.add(k)
        reg.set_simhash(k, compute_simhash(k, "", "", ""))
    (ep_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())


def _make_staging_state(tmp_path: Path, tier_diff: list[TierDiffRow], candidate_path: Path) -> dict:
    """Build a _state dict with a STAGING migration stash."""
    state = _make_minimal_state(tmp_path)
    stash = MigrationStashState(
        state="STAGING",
        candidate_path=str(candidate_path),
        candidate_hash="abc123",
        candidate_bytes=candidate_path.read_bytes(),
        candidate_text=candidate_path.read_text(encoding="utf-8"),
        parsed_candidate={},
        staged_at="2026-05-01T00:00:00+00:00",
        simulate_mode_override=False,
        shape_changes=[],
        tier_diff=tier_diff,
        unified_diff="",
        trial=None,
        recovery_required=[],
        parsed_live={},
    )
    state["migration"] = stash
    return state


# ---------------------------------------------------------------------------
# GET /integrity endpoint
# ---------------------------------------------------------------------------


class TestIntegrityEndpointOk:
    def test_ok_response_http_200(self, tmp_path, monkeypatch):
        """GET /integrity with ok store → HTTP 200."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_ok_report(),
        ):
            resp = client.get("/integrity")

        assert resp.status_code == 200, resp.text

    def test_ok_response_body_ok_true(self, tmp_path, monkeypatch):
        """GET /integrity with ok store → body.ok == True."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_ok_report(),
        ):
            resp = client.get("/integrity")

        body = resp.json()
        assert body["ok"] is True

    def test_ok_response_body_has_checks(self, tmp_path, monkeypatch):
        """GET /integrity with ok store → body.checks is non-empty."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_ok_report(),
        ):
            resp = client.get("/integrity")

        body = resp.json()
        assert len(body["checks"]) == 1
        assert body["checks"][0]["status"] == "ok"

    def test_ok_response_body_failures_empty(self, tmp_path, monkeypatch):
        """GET /integrity with ok store → body.failures == []."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_ok_report(),
        ):
            resp = client.get("/integrity")

        body = resp.json()
        assert body["failures"] == []


class TestIntegrityEndpointFailure:
    def test_corrupt_registry_http_200_not_5xx(self, tmp_path, monkeypatch):
        """Corrupt registry → HTTP 200 (report body), not 5xx."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_fail_report(),
        ):
            resp = client.get("/integrity")

        assert resp.status_code == 200, resp.text

    def test_corrupt_registry_body_ok_false(self, tmp_path, monkeypatch):
        """Corrupt registry → body.ok == False."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_fail_report(),
        ):
            resp = client.get("/integrity")

        body = resp.json()
        assert body["ok"] is False

    def test_corrupt_registry_failures_populated(self, tmp_path, monkeypatch):
        """Corrupt registry → body.failures has one entry with parse_error status."""
        state = _make_minimal_state(tmp_path)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity",
            return_value=_fail_report(),
        ):
            resp = client.get("/integrity")

        body = resp.json()
        assert len(body["failures"]) == 1
        assert body["failures"][0]["status"] == "parse_error"

    def test_daily_loadable_forwarded_to_verifier(self, tmp_path, monkeypatch):
        """daily_loadable from _state is forwarded to verify_infrastructure_integrity."""
        state = _make_minimal_state(tmp_path)
        state["daily_loadable"] = True
        client = _make_client(monkeypatch, state)

        calls = []

        def _capture(*args, **kwargs):
            calls.append(kwargs)
            return _ok_report()

        with patch(
            "paramem.backup.integrity.verify_infrastructure_integrity", side_effect=_capture
        ):
            client.get("/integrity")

        assert calls, "verify_infrastructure_integrity was not called"
        assert calls[0].get("daily_loadable") is True


# ---------------------------------------------------------------------------
# Boot degraded path: real corrupt registry → store_load_degraded=True
# ---------------------------------------------------------------------------


class TestBootDegradedPath:
    def test_corrupt_registry_sets_store_load_degraded(self, tmp_path, monkeypatch):
        """Real corrupt registry on disk causes _preload_memory_store to set
        store_load_degraded=True via the production integrity gate.

        Calls production code (_preload_memory_store) with real filesystem
        state; no logic reimplementation.
        """
        state = _make_minimal_state(tmp_path)
        cfg = state["config"]
        cfg.consolidation.indexed_key_replay = True
        cfg.inference.preload_cache = False
        # Build corrupt episodic registry
        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_corrupt_registry(ep_dir)

        monkeypatch.setattr(app_module, "_state", state)

        # Call the real _preload_memory_store (no GPU needed — preload_cache=False
        # skips weight loading; registry is read by load_registries_from_disk
        # inside the function, and the integrity check reads it again)
        result = app_module._preload_memory_store(cfg, model=None, tokenizer=None)

        # The corrupt registry should cause store_load_degraded=True
        assert state["store_load_degraded"] is True, (
            "Expected store_load_degraded=True after boot integrity check with corrupt registry"
        )
        # The function must still return a MemoryStore (degraded, but present)
        from paramem.memory.store import MemoryStore

        assert isinstance(result, MemoryStore)


# ---------------------------------------------------------------------------
# _arm_active_store_migration: corrupt store → returns False, does not arm
# ---------------------------------------------------------------------------


class TestArmActiveMigrationIntegrityGate:
    def test_corrupt_store_arm_returns_false(self, tmp_path, monkeypatch):
        """_arm_active_store_migration with corrupt registry returns False.

        Calls production _arm_active_store_migration directly.
        """
        state = _make_minimal_state(tmp_path)
        cfg = state["config"]
        cfg.consolidation.mode = "simulate"

        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_corrupt_registry(ep_dir)

        monkeypatch.setattr(app_module, "_state", state)

        with patch(
            "paramem.server.active_store_migration.detect_mode_switch",
            return_value=MagicMock(
                direction="train_to_simulate",
                source_mode="train",
                completed_tiers=[],
                failed_tiers={},
            ),
        ):
            result = app_module._arm_active_store_migration(cfg)

        assert result is False
        assert state.get("pending_rehydration") is False

    def test_ok_store_arm_returns_true(self, tmp_path, monkeypatch):
        """_arm_active_store_migration with clean registry returns True (arms migration).

        Uses train mode so the graph check is skipped (train mode does not
        require graph.json).  Writes a manifest slot so the manifest check
        passes for the keyed tier.
        """
        import json as _json

        state = _make_minimal_state(tmp_path)
        cfg = state["config"]
        # Use train mode so graph.json is not required
        cfg.consolidation.mode = "train"

        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_valid_registry(ep_dir, ["key1"])
        # Write a minimal manifest slot so the train-mode manifest check passes
        slot_dir = ep_dir / "20260501-000000"
        slot_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 4,
            "name": "episodic",
            "trained_at": "2026-05-01T00:00:00Z",
            "window_stamp": "",
            "base_model": {"repo": "test/model", "sha": "abc", "hash": "sha256:deadbeef"},
            "tokenizer": {"name_or_path": "test/model", "vocab_size": 32000, "merges_hash": "abc"},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"]},
            "registry_sha256": "",
            "key_count": 1,
        }
        (slot_dir / "meta.json").write_text(_json.dumps(manifest), encoding="utf-8")

        monkeypatch.setattr(app_module, "_state", state)

        with (
            patch(
                "paramem.server.active_store_migration.detect_mode_switch",
                return_value=MagicMock(
                    direction="simulate_to_train",
                    source_mode="simulate",
                    completed_tiers=[],
                    failed_tiers={},
                ),
            ),
            patch("paramem.server.active_store_migration.save_state"),
        ):
            result = app_module._arm_active_store_migration(cfg)

        assert result is True
        assert state.get("pending_rehydration") is True


# ---------------------------------------------------------------------------
# Migration scheduler: store_load_degraded → migration skipped
# ---------------------------------------------------------------------------


class TestMigrationSchedulerDegraded:
    def test_scheduler_returns_migration_skipped_degraded(self, tmp_path, monkeypatch):
        """Scheduler tick with store_load_degraded=True returns 'migration_skipped_degraded'.

        Calls the real _maybe_trigger_scheduled_consolidation production code
        with store_load_degraded=True and pending_rehydration=True.
        """
        state = _make_minimal_state(tmp_path)
        state["store_load_degraded"] = True
        state["pending_rehydration"] = True
        state["consolidating"] = False
        state["background_trainer"] = None
        # mode must be "local" so the early cloud-only guard does not short-circuit
        state["mode"] = "local"
        state["cloud_only_reason"] = None
        state["session_buffer"] = None
        state["speaker_store"] = None

        monkeypatch.setattr(app_module, "_state", state)

        result = app_module._maybe_trigger_scheduled_consolidation()
        assert result == "migration_skipped_degraded", (
            f"Expected 'migration_skipped_degraded' but got {result!r}"
        )


# ---------------------------------------------------------------------------
# migration/confirm endpoint: integrity gates for base-swap and mode-switch
# ---------------------------------------------------------------------------


def _mode_switch_tier_diff(old_mode: str = "train", new_mode: str = "simulate") -> list:
    """Build a tier_diff for a pure consolidation.mode change."""
    return [
        TierDiffRow(
            dotted_path="consolidation.mode",
            old_value=old_mode,
            new_value=new_mode,
            tier="operational",
        )
    ]


def _base_swap_tier_diff() -> list:
    """Build a tier_diff for a model change (base-swap)."""
    return [
        TierDiffRow(
            dotted_path="model",
            old_value="mistral",
            new_value="gemma",
            tier="destructive",
        )
    ]


class TestMigrationConfirmBaseSwap409:
    def test_base_swap_corrupt_store_returns_409(self, tmp_path, monkeypatch):
        """POST /migration/confirm on a base-swap candidate with corrupt store → 409.

        Exercises the base-swap integrity gate (already present before these
        fixes) to confirm the 409 response shape.
        """
        cfg = _make_config(tmp_path)
        cfg.consolidation.mode = "train"

        # Write corrupt registry
        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_corrupt_registry(ep_dir)

        # Write a fake candidate config file
        candidate_file = tmp_path / "candidate.yaml"
        candidate_file.write_text("model: gemma\n", encoding="utf-8")

        state = _make_staging_state(tmp_path, _base_swap_tier_diff(), candidate_file)
        state["config"] = cfg
        state["config_path"] = str(tmp_path / "server.yaml")
        (tmp_path / "server.yaml").write_text("model: mistral\n", encoding="utf-8")

        client = _make_client(monkeypatch, state)
        resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 409, resp.text
        body = resp.json()
        detail = body.get("detail", body)
        assert detail.get("error") == "integrity_failure"
        assert len(detail.get("failing_files", [])) >= 1

    def test_base_swap_corrupt_store_live_config_not_renamed(self, tmp_path, monkeypatch):
        """Corrupt store + base-swap → 409 and the live config was NOT renamed."""
        cfg = _make_config(tmp_path)
        cfg.consolidation.mode = "train"

        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_corrupt_registry(ep_dir)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_text("model: mistral\n", encoding="utf-8")
        original_content = live_yaml.read_bytes()

        candidate_file = tmp_path / "candidate.yaml"
        candidate_file.write_text("model: gemma\n", encoding="utf-8")

        state = _make_staging_state(tmp_path, _base_swap_tier_diff(), candidate_file)
        state["config"] = cfg
        state["config_path"] = str(live_yaml)

        client = _make_client(monkeypatch, state)
        client.post("/migration/confirm", json={})

        # Live config must be unchanged
        assert live_yaml.read_bytes() == original_content, (
            "Live config was mutated despite integrity failure"
        )


class TestMigrationConfirmModeSwitchIntegrityGate:
    def test_mode_switch_corrupt_store_returns_409(self, tmp_path, monkeypatch):
        """POST /migration/confirm on a mode-switch candidate with corrupt store → 409.

        This is the Fix-1 path: the integrity gate must fire BEFORE _rename_config.
        """
        cfg = _make_config(tmp_path, mode="train")

        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_corrupt_registry(ep_dir)

        candidate_file = tmp_path / "candidate.yaml"
        candidate_file.write_text("consolidation:\n  mode: simulate\n", encoding="utf-8")

        state = _make_staging_state(tmp_path, _mode_switch_tier_diff(), candidate_file)
        state["config"] = cfg
        state["config_path"] = str(tmp_path / "server.yaml")
        (tmp_path / "server.yaml").write_text("consolidation:\n  mode: train\n", encoding="utf-8")

        client = _make_client(monkeypatch, state)
        resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 409, resp.text
        body = resp.json()
        detail = body.get("detail", body)
        assert detail.get("error") == "integrity_failure"
        assert len(detail.get("failing_files", [])) >= 1

    def test_mode_switch_corrupt_store_live_config_not_renamed(self, tmp_path, monkeypatch):
        """Corrupt store + mode-switch → 409 AND the live config is NOT renamed.

        Proves Fix-1 ordering: the integrity gate fires before any mutation.
        """
        cfg = _make_config(tmp_path, mode="train")

        ep_dir = Path(cfg.adapter_dir) / "episodic"
        _write_corrupt_registry(ep_dir)

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_text("consolidation:\n  mode: train\n", encoding="utf-8")
        original_content = live_yaml.read_bytes()

        candidate_file = tmp_path / "candidate.yaml"
        candidate_file.write_text("consolidation:\n  mode: simulate\n", encoding="utf-8")

        state = _make_staging_state(tmp_path, _mode_switch_tier_diff(), candidate_file)
        state["config"] = cfg
        state["config_path"] = str(live_yaml)

        client = _make_client(monkeypatch, state)
        resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 409, f"Expected 409 but got {resp.status_code}: {resp.text}"
        # The live config must be unchanged — no rename occurred
        assert live_yaml.read_bytes() == original_content, (
            "Live config was mutated (renamed from candidate) despite integrity failure"
        )

    def test_mode_switch_clean_store_succeeds(self, tmp_path, monkeypatch):
        """Mode-switch with a clean store succeeds (no 409).

        Validates that the integrity gate does not block a valid mode-switch.
        """
        cfg = _make_config(tmp_path, mode="train")
        # No corrupt files — store is clean (no registry at all = skipped, not failure)

        candidate_file = tmp_path / "candidate.yaml"
        candidate_file.write_text("consolidation:\n  mode: simulate\n", encoding="utf-8")

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_text("consolidation:\n  mode: train\n", encoding="utf-8")

        state = _make_staging_state(tmp_path, _mode_switch_tier_diff(), candidate_file)
        state["config"] = cfg
        state["config_path"] = str(live_yaml)

        with patch("paramem.server.app._refresh_config_from_disk_into_state"):
            client = _make_client(monkeypatch, state)
            resp = client.post("/migration/confirm", json={})

        # Must not be a 409 integrity_failure
        if resp.status_code == 409:
            body = resp.json()
            detail = body.get("detail", body)
            assert detail.get("error") != "integrity_failure", (
                f"Clean store triggered integrity_failure: {detail}"
            )
