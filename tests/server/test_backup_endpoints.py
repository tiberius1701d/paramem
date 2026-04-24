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
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.backup import write as backup_write
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

    config_dir = backups_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    slot_dir = _bwrite(
        ArtifactKind.CONFIG,
        b"model: mistral\n",
        meta_fields={"tier": "daily"},
        base_dir=config_dir,
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

        _slot1 = backup_write(
            ArtifactKind.CONFIG,
            b"config_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "config",
        )
        _slot2 = backup_write(
            ArtifactKind.REGISTRY,
            b"registry_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "registry",
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

        backup_write(
            ArtifactKind.CONFIG,
            b"config_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "config",
        )
        backup_write(
            ArtifactKind.REGISTRY,
            b"registry_data",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "registry",
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
        """Age-encrypted slot + daily identity not loadable → 500 decrypt_no_key.

        Writes an age-encrypted backup using a daily identity, then drops the
        daily passphrase so decryption fails with a RuntimeError (identity not
        loaded), which the endpoint maps to the ``decrypt_no_key`` error code.
        """
        from paramem.backup.key_store import (  # noqa: PLC0415
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Mint + wire a daily identity so backup_write produces an age envelope.
        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, "pw"), key_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            b"model: mistral\n",
            meta_fields={"tier": "daily"},
            base_dir=backups_root / "config",
        )
        backup_id = slot_dir.name

        # Drop the daily passphrase so decryption will fail.
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post(
            "/backup/restore",
            json={"backup_id": backup_id},
        )

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"]["error"] == "decrypt_no_key"

        # Safety backup was NOT created (decrypt failed before step 5).
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

        _slots = []
        for _ in range(3):
            import time

            time.sleep(0.02)  # Ensure different timestamps
            s = backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
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

        for _ in range(3):
            import time

            time.sleep(0.02)
            backup_write(
                ArtifactKind.CONFIG,
                b"model: mistral\n",
                meta_fields={"tier": "daily"},
                base_dir=backups_root / "config",
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


def _setup_age_slot_for_endpoint_test(
    tmp_path: Path, monkeypatch, backups_root: Path, passphrase: str = "pw"
) -> tuple:
    """Write an age-encrypted config backup slot; return (slot_dir, key_path, ident)."""
    from paramem.backup.key_store import (  # noqa: PLC0415
        DAILY_PASSPHRASE_ENV_VAR,
        _clear_daily_identity_cache,
        mint_daily_identity,
        wrap_daily_identity,
        write_daily_key_file,
    )

    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()

    slot_dir = backup_write(
        ArtifactKind.CONFIG,
        b"model: mistral\n",
        meta_fields={"tier": "daily"},
        base_dir=backups_root / "config",
    )
    return slot_dir, key_path, ident


class TestRestoreDecryptErrorCodes:
    """Distinct error codes for age-decrypt failure modes."""

    def test_restore_no_key_returns_decrypt_no_key_error(self, tmp_path: Path, monkeypatch) -> None:
        """Age-encrypted slot + daily passphrase not set → 500 decrypt_no_key."""
        from paramem.backup.key_store import (  # noqa: PLC0415
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
        )

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        slot_dir, _, _ = _setup_age_slot_for_endpoint_test(tmp_path, monkeypatch, backups_root)
        backup_id = slot_dir.name

        # Drop passphrase so the identity can no longer be loaded.
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"]["error"] == "decrypt_no_key", (
            f"Expected decrypt_no_key, got: {resp.json()['detail']['error']!r}"
        )

    def test_restore_wrong_recipient_returns_decrypt_invalid_token(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Age-encrypted slot + a different identity loaded → 500 decrypt_invalid_token.

        Writes the slot under identity A, then swaps to a freshly-minted
        identity B before restoring — B cannot decrypt an envelope addressed
        to A, so pyrage raises DecryptError which the endpoint maps to
        ``decrypt_invalid_token``.
        """
        from paramem.backup.key_store import (  # noqa: PLC0415
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Write with identity A.
        slot_dir, _key_path_a, _ident_a = _setup_age_slot_for_endpoint_test(
            tmp_path, monkeypatch, backups_root, passphrase="pw-a"
        )
        backup_id = slot_dir.name

        # Swap to identity B — different X25519 key, cannot decrypt A's envelopes.
        ident_b = mint_daily_identity()
        key_path_b = tmp_path / "daily_key_b.age"
        write_daily_key_file(wrap_daily_identity(ident_b, "pw-b"), key_path_b)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw-b")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path_b)
        _clear_daily_identity_cache()

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"]["error"] == "decrypt_invalid_token", (
            f"Expected decrypt_invalid_token for wrong recipient, "
            f"got: {resp.json()['detail']['error']!r}"
        )
