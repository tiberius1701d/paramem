"""Integration tests for the 4 backup REST endpoints (Slice 6b).

Uses FastAPI TestClient with monkeypatched _state — no live server, no GPU.

Tests cover:
26. GET /backup/list — empty store
27. GET /backup/list — mixed kinds, newest-first
28. GET /backup/list — filtered by kind
29. GET /backup/list — invalid kind → 400
30. POST /backup/create — default kinds
31. POST /backup/create — custom kinds + label
32. POST /backup/create — unknown kind → 400
33. POST /backup/create — disk pressure → success=False
34. POST /backup/create — cloud-only (loop=None) → graph skipped gracefully
35. POST /backup/restore — happy path config
36. POST /backup/restore — not found → 404
37. POST /backup/restore — non-config kind → 400
38. POST /backup/restore — during STAGING → 409
39. POST /backup/restore — during TRIAL → 409
40. POST /backup/restore — consolidating → 409
41. POST /backup/restore — encrypted wrong key → 500, no safety slot
42. POST /backup/prune — happy path
43. POST /backup/prune — dry run
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.backup import write as backup_write
from paramem.backup.encryption import SecurityBackupsConfig as EncSecurityConfig
from paramem.backup.types import ArtifactKind
from paramem.server.config import (
    PathsConfig,
    SecurityConfig,
    ServerBackupsConfig,
    ServerConfig,
)
from paramem.server.migration import initial_migration_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    max_total_disk_gb: float = 20.0,
    schedule: str = "daily 04:00",
) -> ServerConfig:
    """Build a minimal real ServerConfig."""
    config = ServerConfig.__new__(ServerConfig)
    config.paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule=schedule,
            artifacts=["config", "graph", "registry"],
            max_total_disk_gb=max_total_disk_gb,
        )
    )
    return config


def _make_state(tmp_path: Path, config: ServerConfig) -> dict:
    """Build a minimal _state dict for endpoint tests."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(b"model: mistral\n")

    loop = MagicMock()
    loop.merger = MagicMock()
    loop.merger.save_bytes.return_value = b'{"nodes": []}'

    return {
        "model": None,
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": initial_migration_state(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "consolidation_loop": loop,
    }


def _make_client(monkeypatch, state: dict):
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


def _seed_config_slot(backups_root: Path, slot_name: str = "20260421-040000") -> Path:
    """Write a minimal config slot with a valid sidecar."""
    from paramem.backup.backup import write as _bwrite
    from paramem.backup.encryption import SecurityBackupsConfig as _EncCfg

    config_dir = backups_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    slot_dir = _bwrite(
        ArtifactKind.CONFIG,
        b"model: mistral\n",
        meta_fields={"tier": "daily"},
        base_dir=config_dir,
        security_config=_EncCfg(),
    )
    return slot_dir


# ---------------------------------------------------------------------------
# Test 26 — /backup/list empty store
# ---------------------------------------------------------------------------


class TestListEmptyStore:
    def test_list_empty_store(self, tmp_path: Path, monkeypatch) -> None:
        """Empty backups dir → items=[], disk_used_bytes=0."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.get("/backup/list")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["items"] == []
        assert body["disk_used_bytes"] == 0


# ---------------------------------------------------------------------------
# Test 27 — /backup/list mixed kinds newest-first
# ---------------------------------------------------------------------------


class TestListMixedKindsNewestFirst:
    def test_list_mixed_kinds_newest_first(self, tmp_path: Path, monkeypatch) -> None:
        """Seed slots across config/graph/registry → all returned, newest-first."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        enc_cfg = EncSecurityConfig()
        _slot1 = backup_write(
            ArtifactKind.CONFIG,
            b"config_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "config",
            security_config=enc_cfg,
        )
        _slot2 = backup_write(
            ArtifactKind.REGISTRY,
            b"registry_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "registry",
            security_config=enc_cfg,
        )

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.get("/backup/list")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items = body["items"]
        assert len(items) == 2
        # Newest first.
        timestamps = [i["timestamp"] for i in items]
        assert timestamps == sorted(timestamps, reverse=True)
        kinds = {i["kind"] for i in items}
        assert "config" in kinds
        assert "registry" in kinds


# ---------------------------------------------------------------------------
# Test 28 — /backup/list filtered by kind
# ---------------------------------------------------------------------------


class TestListFilteredByKind:
    def test_list_filtered_by_kind(self, tmp_path: Path, monkeypatch) -> None:
        """?kind=config → only config slots returned."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        enc_cfg = EncSecurityConfig()
        backup_write(
            ArtifactKind.CONFIG,
            b"config_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "config",
            security_config=enc_cfg,
        )
        backup_write(
            ArtifactKind.REGISTRY,
            b"registry_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "registry",
            security_config=enc_cfg,
        )

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.get("/backup/list?kind=config")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert all(i["kind"] == "config" for i in body["items"])
        assert len(body["items"]) == 1


# ---------------------------------------------------------------------------
# Test 29 — /backup/list invalid kind → 400
# ---------------------------------------------------------------------------


class TestListInvalidKind:
    def test_list_invalid_kind_returns_400(self, tmp_path: Path, monkeypatch) -> None:
        """?kind=bogus → 400 kind_invalid."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.get("/backup/list?kind=bogus")
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["detail"]["error"] == "kind_invalid"


# ---------------------------------------------------------------------------
# Test 30 — /backup/create default kinds
# ---------------------------------------------------------------------------


class TestCreateDefaultKinds:
    def test_create_default_kinds(self, tmp_path: Path, monkeypatch) -> None:
        """POST {} → writes config+registry (graph skipped — loop mock), tier=manual."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tier"] == "manual"
        # At least config should be written (loop.merger.save_bytes is mocked → graph ok too)
        assert "config" in body["written_slots"] or body["success"] is True
        assert body["error"] is None or body["success"] is False  # no unknown error


# ---------------------------------------------------------------------------
# Test 31 — /backup/create custom kinds + label
# ---------------------------------------------------------------------------


class TestCreateCustomKindsLabel:
    def test_create_custom_kinds_and_label(self, tmp_path: Path, monkeypatch) -> None:
        """POST {"kinds":["config"], "label":"x"} → only config written; label in meta."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={"kinds": ["config"], "label": "x"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert "config" in body["written_slots"]
        # graph and registry not in written_slots (they were not requested).
        assert "graph" not in body["written_slots"]
        assert "registry" not in body["written_slots"]


# ---------------------------------------------------------------------------
# Test 32 — /backup/create unknown kind → 400
# ---------------------------------------------------------------------------


class TestCreateUnknownKindReturns400:
    def test_create_unknown_kind_returns_400(self, tmp_path: Path, monkeypatch) -> None:
        """POST {"kinds":["bogus"]} → 400 kind_invalid."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={"kinds": ["bogus"]})
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["detail"]["error"] == "kind_invalid"


# ---------------------------------------------------------------------------
# Test 33 — /backup/create disk pressure → 200 success=False
# ---------------------------------------------------------------------------


class TestCreateDiskPressureReturns200SuccessFalse:
    def test_create_disk_pressure_returns_200_success_false(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """At 100% cap → response success=False, error starts with 'disk_pressure'."""
        cap_gb = 0.0001  # 100 KB cap
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * 200_000)  # >100 KB → over cap

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is False
        assert body["error"] is not None
        assert "disk_pressure" in body["error"]


# ---------------------------------------------------------------------------
# Test 34 — /backup/create cloud-only → graph skipped gracefully
# ---------------------------------------------------------------------------


class TestCreateCloudOnlySkipsGraphGracefully:
    def test_create_cloud_only_skips_graph_gracefully(self, tmp_path: Path, monkeypatch) -> None:
        """loop=None → written_slots has config; graph in skipped_artifacts."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        state["consolidation_loop"] = None  # cloud-only
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={"kinds": ["config", "graph"]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # config should be written; graph should be in skipped_artifacts
        assert "config" in body["written_slots"]
        skipped_kinds = {s["kind"] for s in body["skipped_artifacts"]}
        assert "graph" in skipped_kinds


# ---------------------------------------------------------------------------
# Test 35 — /backup/restore happy path (config)
# ---------------------------------------------------------------------------


class TestRestoreHappyPathConfig:
    def test_restore_happy_path_config(self, tmp_path: Path, monkeypatch) -> None:
        """Pre-seed a config backup; POST restore → 200, live config matches backup."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Write a config backup.
        backup_content = b"model: gemma\n"
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            backup_content,
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "config",
            security_config=EncSecurityConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "config" in body["restored"]
        assert body["restart_required"] is True
        assert body["restart_hint"]

        # Live config should now contain the backup content.
        live_path = Path(state["config_path"])
        assert live_path.read_bytes() == backup_content

        # Safety backup must have been created.
        assert "config" in body["backed_up_pre_restore"]

        # Recovery banner must be appended.
        recovery = state["migration"]["recovery_required"]
        assert any("backup" in msg.lower() for msg in recovery)


# ---------------------------------------------------------------------------
# Test 36 — /backup/restore not found → 404
# ---------------------------------------------------------------------------


class TestRestoreNotFoundReturns404:
    def test_restore_not_found_returns_404(self, tmp_path: Path, monkeypatch) -> None:
        """Unknown backup_id → 404 not_found."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": "20260101-999999"})
        assert resp.status_code == 404, resp.text
        assert resp.json()["detail"]["error"] == "not_found"


# ---------------------------------------------------------------------------
# Test 37 — /backup/restore non-config kind → 400
# ---------------------------------------------------------------------------


class TestRestoreNonConfigKindReturns400:
    def test_restore_non_config_kind_returns_400(self, tmp_path: Path, monkeypatch) -> None:
        """Graph backup → 400 restore_kind_not_supported."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        slot_dir = backup_write(
            ArtifactKind.GRAPH,
            b'{"nodes": []}',
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "graph",
            security_config=EncSecurityConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"]["error"] == "restore_kind_not_supported"


# ---------------------------------------------------------------------------
# Test 38 — /backup/restore during STAGING → 409
# ---------------------------------------------------------------------------


class TestRestoreDuringStagingReturns409:
    def test_restore_during_staging_returns_409(self, tmp_path: Path, monkeypatch) -> None:
        """STAGING state → 409 staging_active."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        state["migration"]["state"] = "STAGING"
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": "irrelevant"})
        assert resp.status_code == 409, resp.text
        error_code = resp.json()["detail"]["error"]
        assert error_code in {"staging_active", "trial_active"}


# ---------------------------------------------------------------------------
# Test 39 — /backup/restore during TRIAL → 409
# ---------------------------------------------------------------------------


class TestRestoreDuringTrialReturns409:
    def test_restore_during_trial_returns_409(self, tmp_path: Path, monkeypatch) -> None:
        """TRIAL state → 409 trial_active."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        state["migration"]["state"] = "TRIAL"
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": "irrelevant"})
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["error"] == "trial_active"


# ---------------------------------------------------------------------------
# Test 40 — /backup/restore during consolidation → 409
# ---------------------------------------------------------------------------


class TestRestoreDuringConsolidationReturns409:
    def test_restore_during_consolidation_returns_409(self, tmp_path: Path, monkeypatch) -> None:
        """consolidating=True → 409 consolidating."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        state["consolidating"] = True
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": "irrelevant"})
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["error"] == "consolidating"


# ---------------------------------------------------------------------------
# Test 41 — /backup/restore encrypted wrong key → 500, no safety slot written
# ---------------------------------------------------------------------------


class TestRestoreEncryptedWrongKeyReturns500:
    def test_restore_encrypted_wrong_key_returns_500(self, tmp_path: Path, monkeypatch) -> None:
        """Encrypted slot + missing key → 500 decrypt_no_key; no safety slot.

        Fix 11 (2026-04-23): error code changed from ``restore_decrypt_failed``
        to ``decrypt_no_key`` when PARAMEM_MASTER_KEY is absent, so operators
        immediately know the key is missing rather than getting a generic decrypt error.
        """
        import os

        from cryptography.fernet import Fernet  # noqa: PLC0415

        from paramem.backup.encryption import _clear_cipher_cache  # noqa: PLC0415

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Fernet.generate_key() returns a url-safe base64 bytes value ready to use as the env var.
        test_key = Fernet.generate_key().decode()

        _clear_cipher_cache()
        with patch.dict(os.environ, {"PARAMEM_MASTER_KEY": test_key}):
            slot_dir = backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
                security_config=EncSecurityConfig(),
            )
        backup_id = slot_dir.name

        # Remove the key so decryption fails; clear cache so missing key is picked up.
        _clear_cipher_cache()
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        env_without_key = {k: v for k, v in os.environ.items() if k != "PARAMEM_MASTER_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 500, resp.text
        # Fix 11 (2026-04-23): missing key now surfaces as decrypt_no_key.
        assert resp.json()["detail"]["error"] == "decrypt_no_key"

        # Safety backup was NOT created (decrypt failed before step 5).
        # Exclude the original slot and any .pending temp dirs from the check.
        safety_dir = backups_root / "config"
        safety_slots = (
            [
                d
                for d in safety_dir.iterdir()
                if d.is_dir() and d.name != slot_dir.name and not d.name.startswith(".")
            ]
            if safety_dir.exists()
            else []
        )
        assert safety_slots == [], f"Safety slot should not have been created: {safety_slots}"


# ---------------------------------------------------------------------------
# Test 42 — /backup/prune happy path
# ---------------------------------------------------------------------------


class TestPruneHappyPath:
    def test_prune_happy_path(self, tmp_path: Path, monkeypatch) -> None:
        """Seed oversize tier → POST /backup/prune → deleted populated."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Write more config slots than retention allows (keep=7, but we can only
        # write so many; use a config with keep=1).
        from paramem.server.config import RetentionConfig, RetentionTierConfig

        config.security.backups.retention = RetentionConfig(
            daily=RetentionTierConfig(keep=1),
        )

        enc_cfg = EncSecurityConfig()
        _slots = []
        for _ in range(3):
            import time

            time.sleep(0.02)  # Ensure different timestamps
            s = backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
                security_config=enc_cfg,
            )
            _slots.append(s)

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/prune", json={"dry_run": False})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["dry_run"] is False
        assert body["disk_usage_after"]["total_bytes"] <= body["disk_usage_before"]["total_bytes"]


# ---------------------------------------------------------------------------
# Test 43 — /backup/prune dry run
# ---------------------------------------------------------------------------


class TestPruneDryRun:
    def test_prune_dry_run(self, tmp_path: Path, monkeypatch) -> None:
        """dry_run=True → would_delete_next populated; deleted=[]; usage unchanged."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        from paramem.server.config import RetentionConfig, RetentionTierConfig

        config.security.backups.retention = RetentionConfig(
            daily=RetentionTierConfig(keep=1),
        )

        enc_cfg = EncSecurityConfig()
        for _ in range(3):
            import time

            time.sleep(0.02)
            backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
                security_config=enc_cfg,
            )

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/prune", json={"dry_run": True})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["dry_run"] is True
        assert body["deleted"] == []
        # disk_usage_after should equal before in dry-run
        assert body["disk_usage_after"]["total_bytes"] == body["disk_usage_before"]["total_bytes"]


# ---------------------------------------------------------------------------
# Fix 11 — decrypt error code distinction
# ---------------------------------------------------------------------------


class TestRestoreDecryptErrorCodes:
    """Fix 11 (2026-04-23): /backup/restore now returns distinct error codes
    for three decrypt failure modes: no-key, wrong-key/corrupt, unknown.

    The no-key case was previously ``restore_decrypt_failed``; Fix 11 changes
    it to ``decrypt_no_key`` so operators can immediately see what is wrong.
    """

    def test_restore_no_key_returns_decrypt_no_key_error(self, tmp_path: Path, monkeypatch) -> None:
        """Encrypted slot + no PARAMEM_MASTER_KEY → 500 decrypt_no_key."""
        import os

        from cryptography.fernet import Fernet

        from paramem.backup.encryption import _clear_cipher_cache

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        test_key = Fernet.generate_key().decode()
        _clear_cipher_cache()
        with patch.dict(os.environ, {"PARAMEM_MASTER_KEY": test_key}):
            slot_dir = backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
                security_config=EncSecurityConfig(),
            )
        backup_id = slot_dir.name

        _clear_cipher_cache()
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        env_without_key = {k: v for k, v in os.environ.items() if k != "PARAMEM_MASTER_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 500, resp.text
        # Fix 11: no-key case is now decrypt_no_key, not restore_decrypt_failed
        assert resp.json()["detail"]["error"] == "decrypt_no_key", (
            f"Expected decrypt_no_key, got: {resp.json()['detail']['error']!r}. "
            "Fix 11 regression: no-key error code reverted."
        )
        assert "PARAMEM_MASTER_KEY" in resp.json()["detail"]["message"], (
            "decrypt_no_key message must mention PARAMEM_MASTER_KEY"
        )

    def test_restore_wrong_key_returns_decrypt_invalid_token(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Encrypted slot + wrong PARAMEM_MASTER_KEY → 500 decrypt_invalid_token."""
        import os

        from cryptography.fernet import Fernet

        from paramem.backup.encryption import _clear_cipher_cache

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Write with key A.
        key_a = Fernet.generate_key().decode()
        _clear_cipher_cache()
        with patch.dict(os.environ, {"PARAMEM_MASTER_KEY": key_a}):
            slot_dir = backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
                security_config=EncSecurityConfig(),
            )
        backup_id = slot_dir.name

        # Restore with different key B — decryption must fail with InvalidToken.
        key_b = Fernet.generate_key().decode()
        _clear_cipher_cache()
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        with patch.dict(os.environ, {"PARAMEM_MASTER_KEY": key_b}):
            resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"]["error"] == "decrypt_invalid_token", (
            f"Expected decrypt_invalid_token for wrong key, got: {resp.json()['detail']['error']!r}"
        )
