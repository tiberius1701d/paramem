"""Integration tests for the 4 backup REST endpoints.

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

``restore_bundle`` itself (real bundle/slot fixtures, corrupt-bundle
handling, the incompatible-bundle list-then-refuse arc) is covered in
``tests/backup/test_restore.py`` and ``tests/backup/test_bundle_boot_binding.py``,
not at the REST-endpoint level here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.backup import write as backup_write
from paramem.backup.types import BUNDLE_SCHEMA_VERSION, ArtifactKind
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
            artifacts=["snapshot_bundle"],
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
        backups_root=backups_root,
        backups_cfg=ServerBackupsConfig(),
    )
    return slot_dir


def _seed_restorable_bundle(tmp_path: Path, backups_root: Path) -> str:
    """Write a real, restorable ``snapshot_bundle`` slot: one episodic tier
    carrying a bound simulate slot (registry + key_metadata + graph.json),
    captured through the real ``write_bundle`` path -- the same
    registry-bound-slot shape ``tests/backup/test_bundle_boot_binding.py``
    round-trips through ``restore_bundle``. Returns the backup_id (the
    bundle slot directory name)."""
    import json as _json

    from paramem.backup.backup import write_bundle
    from paramem.training.key_registry import KeyRegistry
    from tests._fold_fixtures import _write_graph

    src = tmp_path / "bundle_src"
    episodic_dir = src / "adapters" / "episodic"
    episodic_dir.mkdir(parents=True, exist_ok=True)

    key = "graph1"
    registry = KeyRegistry()
    registry.add(key)
    registry.set_simhash(key, 12345)
    registry.save(episodic_dir / "indexed_key_registry.json")
    (episodic_dir / "key_metadata.json").write_text(
        _json.dumps(
            {
                "tier_cycle": 0,
                "keys": {
                    key: {
                        "speaker_id": "speaker0",
                        "relation_type": "factual",
                        "reinforcement_count": 1,
                        "last_reinforced_cycle": 0,
                        "last_seen": "2026-01-01T00:00:00Z",
                        "first_seen": "2026-01-01T00:00:00Z",
                        "promoted": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _write_graph(
        episodic_dir,
        [
            {
                "key": key,
                "subject": "alice",
                "predicate": "lives_in",
                "object": "berlin",
                "speaker_id": "speaker0",
            }
        ],
    )

    bundle_slot = write_bundle(
        config_path=tmp_path / "no-such-config.yaml",
        adapter_dirs={"episodic": episodic_dir},
        backups_root=backups_root,
        backups_cfg=None,
        meta_fields={"tier": "manual", "label": "dispose-test-bundle"},
    )
    return bundle_slot.name


# ---------------------------------------------------------------------------
# /backup/list empty store
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
# /backup/list mixed kinds newest-first
# ---------------------------------------------------------------------------


class TestListMixedKindsNewestFirst:
    def test_list_mixed_kinds_newest_first(self, tmp_path: Path, monkeypatch) -> None:
        """Seed slots across config/graph/resume → all returned, newest-first."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        _slot1 = backup_write(
            ArtifactKind.CONFIG,
            b"config_data",
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        _slot2 = backup_write(
            ArtifactKind.RESUME,
            b"resume_data",
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
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
        assert "resume" in kinds


# ---------------------------------------------------------------------------
# /backup/list filtered by kind
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
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_write(
            ArtifactKind.RESUME,
            b"resume_data",
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.get("/backup/list?kind=config")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert all(i["kind"] == "config" for i in body["items"])
        assert len(body["items"]) == 1


# ---------------------------------------------------------------------------
# /backup/list invalid kind → 400
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
# /backup/create default kinds
# ---------------------------------------------------------------------------


class TestCreateDefaultKinds:
    def test_create_default_kinds(self, tmp_path: Path, monkeypatch) -> None:
        """POST {} → default is snapshot_bundle; mock write_bundle → success, tier=manual."""
        from unittest.mock import patch

        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        fake_slot = config.paths.data / "backups" / "snapshot" / "20260521-040000"
        fake_slot.mkdir(parents=True, exist_ok=True)
        (fake_slot / "bundle.meta.json").write_text("{}", encoding="utf-8")

        with patch("paramem.backup.backup.write_bundle", return_value=fake_slot):
            resp = client.post("/backup/create", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tier"] == "manual"
        # Default produces a bundle slot.
        assert "snapshot_bundle" in body["written_slots"] or body["success"] is True


# ---------------------------------------------------------------------------
# /backup/create custom kinds + label
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
        # graph not in written_slots (it was not requested).
        assert "graph" not in body["written_slots"]


# ---------------------------------------------------------------------------
# /backup/create unknown kind → 400
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
# /backup/create honours the tier param (scheduled-timer path)
# ---------------------------------------------------------------------------


class TestCreateTierParam:
    def test_create_tier_daily_files_under_daily(self, tmp_path: Path, monkeypatch) -> None:
        """POST {"tier":"daily"} → slot filed under daily (the timer delegation path)."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={"kinds": ["config"], "tier": "daily"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tier"] == "daily"
        assert body["success"] is True

    def test_create_default_tier_is_manual(self, tmp_path: Path, monkeypatch) -> None:
        """Omitting tier preserves the manual default (operator backups)."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={"kinds": ["config"]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["tier"] == "manual"

    def test_create_invalid_tier_returns_400(self, tmp_path: Path, monkeypatch) -> None:
        """POST {"tier":"bogus"} → 400 tier_invalid."""
        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/create", json={"kinds": ["config"], "tier": "bogus"})
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"]["error"] == "tier_invalid"


# ---------------------------------------------------------------------------
# /backup/create disk pressure → 200 success=False
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
# /backup/create cloud-only → graph skipped gracefully
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
# /backup/restore happy path (config)
# ---------------------------------------------------------------------------


class TestRestoreHappyPathConfig:
    def test_restore_happy_path_config(self, tmp_path: Path, monkeypatch) -> None:
        """Pre-seed a config backup; POST restore → 200, live config matches backup.

        A config-kind restore keeps its existing mechanism (no live-apply
        dispatch, no store quarantine) — the response reports
        ``serving=False`` (an operator restart is still what converges it);
        the response carries no ``restart_required``/``restart_hint`` fields.
        """
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Write a config backup.
        backup_content = b"model: gemma\n"
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            backup_content,
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "config" in body["restored"]
        assert body["serving"] is False
        assert body["quarantine_cause"] is None
        assert "restart_required" not in body
        assert "restart_hint" not in body

        # Live config should now contain the backup content.
        live_path = Path(state["config_path"])
        assert live_path.read_bytes() == backup_content

        # Safety backup must have been created.
        assert "config" in body["backed_up_pre_restore"]

    def test_pending_consolidation_event_config_restore_succeeds_and_leaves_ledger(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A pending consolidation event's stage ledger does NOT refuse a
        config-kind restore either, and — unlike a snapshot_bundle restore —
        it is left untouched: a config restore rewrites no tier, so there is
        nothing for the ledger to misclassify on the next resume."""
        from paramem.training import stage_ledger as sl

        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        backup_content = b"model: gemma\n"
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            backup_content,
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        state_dir = config.paths.data.resolve() / "state"
        scratch = config.paths.data / "adapters" / "episodic" / "cycle_0"
        (scratch / "checkpoint-1").mkdir(parents=True, exist_ok=True)
        ledger = sl.StageLedger(
            version=2,
            event="full",
            venue="weights",
            stamp="20260101T0000",
            tiers={"episodic": {"adapter": "episodic", "pre_sha": "a", "scratch": str(scratch)}},
        )
        sl.write_stages(state_dir, ledger, [])

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "config" in body["restored"]
        # The pending record survives untouched -- a config restore rewrites
        # no tier, so the ledger has nothing to misclassify.
        assert sl.read_ledger(state_dir) is not None
        assert scratch.exists()


class TestPendingConsolidationEventBundleRestoreDisposesLedger:
    def test_pending_consolidation_event_bundle_restore_disposes_ledger_and_resolves_incident(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A snapshot_bundle restore rewrites the episodic tier wholesale, so
        the pending event's stage ledger would misclassify it FOREIGN on the
        next resume -- the record is discarded instead: the
        ledger is disposed (file gone, scratch dir removed) and every
        ``consolidation_resume_blocked`` incident naming it is resolved.

        The real ``restore_bundle`` runs unmocked -- backup_id resolves to a
        genuine bound episodic slot (registry + key_metadata + graph.json)
        via ``write_bundle``, so the endpoint's own dispatch, decrypt-probe,
        and atomic tree rewrite all execute for real. Only the two
        convergence collaborators ``_lift_quarantined_store`` reaches for
        (``_preload_memory_store``, ``QueryRouter``) are patched -- the same
        pair ``TestLiftQuarantinedStore`` patches -- because ``_make_config``
        builds a bare ``ServerConfig`` with no ``adapter_dir`` and the state
        carries no real memory store to rehydrate; the dispose call itself
        is never touched.
        """
        from paramem.server.incidents import read_incidents, record_incident
        from paramem.training import stage_ledger as sl

        config = _make_config(tmp_path)
        # _lift_quarantined_store builds the (patched) QueryRouter's kwargs
        # from config.intent before the call -- Python evaluates
        # keyword-argument expressions unconditionally, so patching
        # QueryRouter alone does not skip that attribute lookup.
        # _make_config's bare ServerConfig (built via __new__, no dataclass
        # defaults) carries no ``intent``; ``adapter_dir`` is a property
        # derived from ``paths.data``, already set, so it needs no help.
        config.intent = None
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        backup_id = _seed_restorable_bundle(tmp_path, backups_root)

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        monkeypatch.setattr(
            app_module, "_preload_memory_store", lambda cfg, *, model, tokenizer: MagicMock()
        )
        monkeypatch.setattr(app_module, "QueryRouter", MagicMock())

        state_dir = config.paths.data.resolve() / "state"
        scratch = config.paths.data / "adapters" / "episodic" / "cycle_0"
        (scratch / "checkpoint-1").mkdir(parents=True, exist_ok=True)
        ledger = sl.StageLedger(
            version=2,
            event="full",
            venue="weights",
            stamp="20260101T0000",
            tiers={"episodic": {"adapter": "episodic", "pre_sha": "a", "scratch": str(scratch)}},
        )
        sl.write_stages(state_dir, ledger, [])
        record_incident(
            state_dir,
            type="consolidation_resume_blocked",
            key="episodic",
            severity="failed",
            summary="Consolidation resume blocked: tier 'episodic' classified 'FOREIGN'",
            detail={"event": "full", "tier": "episodic", "reason": "FOREIGN"},
        )

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "episodic" in body["restored_adapters"]
        assert body["serving"] is True
        assert body["quarantine_cause"] is None

        # The pending record is DISCARDED -- ledger file gone, scratch dir removed.
        assert sl.read_ledger(state_dir) is None
        assert not scratch.exists()

        # Every consolidation_resume_blocked incident naming this tier resolved.
        incidents = read_incidents(state_dir)
        resume_blocked = [i for i in incidents if i.type == "consolidation_resume_blocked"]
        assert resume_blocked, "expected the seeded incident to still be present, now resolved"
        assert all(i.status == "resolved" for i in resume_blocked)


class TestRestoreConfigSafetySlotExemptFromDiskCap:
    def test_restore_config_writes_safety_slot_when_store_over_cap(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The config-branch safety slot is written even when the store is
        already over the configured cap — it is the undo anchor for a config
        restore, deliberately exempt from the cap.  A future edit that
        starts passing a real ``backups_cfg`` at this call site would turn
        this into a 500 ``config_restore_failed``, so this test fails loudly.
        """
        # A cap tiny enough that the seeded backup slot alone exceeds it.
        config = _make_config(tmp_path, max_total_disk_gb=1e-9)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        backup_content = b"model: gemma\n"
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            backup_content,
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=None,  # seed the store without gating the seed itself
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "config" in body["restored"]
        assert "config" in body["backed_up_pre_restore"], (
            "the safety slot must still be written when the store is over cap"
        )


# ---------------------------------------------------------------------------
# /backup/restore — a config-kind backup that cannot be constructed → 400
# ---------------------------------------------------------------------------


class TestRestoreUnbootableConfigReturns400:
    """A backup is a config that was validated against a POSSIBLY OLDER schema.

    New load-time guards (e.g. ``max_interim_count=0`` + ``mode=simulate``) can
    reject bytes that were bootable when the backup was written.  Restoring such a
    backup must be refused — the safety backup is written first so refusing costs
    the operator nothing, and the live config must be left untouched.
    """

    def test_unbootable_config_backup_rejected(self, tmp_path: Path, monkeypatch) -> None:
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        unbootable_content = b"consolidation:\n  mode: simulate\n  max_interim_count: 0\n"
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            unbootable_content,
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)
        live_path = Path(state["config_path"])
        live_bytes_before = live_path.read_bytes()

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"]["error"] == "backup_unbootable"
        assert live_path.read_bytes() == live_bytes_before, "live config was mutated on rejection"

        # The safety backup is written before the construction check — it should
        # exist even though the restore itself was refused.
        config_dir = backups_root / "config"
        slots = [d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) >= 2, "expected the source slot plus a pre_restore safety slot"

    def test_bootable_config_backup_still_restores(self, tmp_path: Path, monkeypatch) -> None:
        """Control: an ordinary bootable backup restores normally."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        backup_content = b"model: gemma\n"
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            backup_content,
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 200, resp.text
        assert Path(state["config_path"]).read_bytes() == backup_content

    def test_a_config_backup_contradicting_the_store_is_rejected_as_unbootable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The same ``validate_candidate`` gate this door already runs also
        runs the config-vs-store check: a backup that constructs cleanly but
        re-points ``paths.data`` at a root holding a populated interim ring
        under a disabled episodic tier is refused as unbootable, and the
        live config is left untouched."""
        config = _make_config(tmp_path)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        candidate_data_root = tmp_path / "restored-data"
        interim_dir = candidate_data_root / "adapters" / "episodic" / "interim_20260101T0000"
        interim_dir.mkdir(parents=True)

        store_contradicting_content = (
            "model: mistral\n"
            "debug: false\n"
            "paths:\n"
            f"  data: {candidate_data_root}\n"
            "consolidation:\n"
            "  max_interim_count: 0\n"
            "adapters:\n"
            "  episodic:\n"
            "    enabled: false\n"
        ).encode()
        slot_dir = backup_write(
            ArtifactKind.CONFIG,
            store_contradicting_content,
            meta_fields={"tier": "daily"},
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)
        live_path = Path(state["config_path"])
        live_bytes_before = live_path.read_bytes()

        resp = client.post("/backup/restore", json={"backup_id": backup_id})

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"]["error"] == "backup_unbootable"
        assert "interim slot(s)" in resp.json()["detail"]["message"]
        assert live_path.read_bytes() == live_bytes_before, "live config was mutated on rejection"


# ---------------------------------------------------------------------------
# /backup/restore not found → 404
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
# /backup/restore non-config kind → 400
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
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
        )
        backup_id = slot_dir.name

        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        resp = client.post("/backup/restore", json={"backup_id": backup_id})
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"]["error"] == "restore_kind_not_supported"


# ---------------------------------------------------------------------------
# /backup/restore during STAGING → 409
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
# /backup/restore during TRIAL → 409
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
# /backup/restore during consolidation → 409
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
# /backup/restore encrypted wrong key → 500, no safety slot written
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
            backups_root=backups_root,
            backups_cfg=ServerBackupsConfig(),
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
# /backup/prune happy path
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
                backups_root=backups_root,
                backups_cfg=ServerBackupsConfig(),
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
# /backup/prune dry run
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
                backups_root=backups_root,
                backups_cfg=ServerBackupsConfig(),
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
# Decrypt error code distinction
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
        backups_root=backups_root,
        backups_cfg=ServerBackupsConfig(),
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


# ---------------------------------------------------------------------------
# /backup/create snapshot_bundle kind
# ---------------------------------------------------------------------------


class TestCreateSnapshotBundleKind:
    """POST /backup/create with kinds=["snapshot_bundle"] routes to write_bundle."""

    def test_snapshot_bundle_kind_accepted(self, tmp_path: Path, monkeypatch) -> None:
        """POST {"kinds":["snapshot_bundle"]} → 200, written_slots has snapshot_bundle."""
        from unittest.mock import patch

        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        # Create a fake bundle slot so write_bundle returns a real path.
        fake_slot = config.paths.data / "backups" / "snapshot" / "20260521-040001"
        fake_slot.mkdir(parents=True, exist_ok=True)
        (fake_slot / "bundle.meta.json").write_text(
            f'{{"bundle_schema_version": {BUNDLE_SCHEMA_VERSION}, "tier": "manual"}}',
            encoding="utf-8",
        )

        with patch("paramem.backup.backup.write_bundle", return_value=fake_slot):
            resp = client.post("/backup/create", json={"kinds": ["snapshot_bundle"]})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert "snapshot_bundle" in body["written_slots"]
        slot_path = Path(body["written_slots"]["snapshot_bundle"])
        assert slot_path.exists()

    def test_snapshot_bundle_tier_daily(self, tmp_path: Path, monkeypatch) -> None:
        """snapshot_bundle with tier=daily → response tier=daily."""
        from unittest.mock import patch

        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        fake_slot = config.paths.data / "backups" / "snapshot" / "20260521-040002"
        fake_slot.mkdir(parents=True, exist_ok=True)
        (fake_slot / "bundle.meta.json").write_text(
            f'{{"bundle_schema_version": {BUNDLE_SCHEMA_VERSION}, "tier": "daily"}}',
            encoding="utf-8",
        )

        with patch("paramem.backup.backup.write_bundle", return_value=fake_slot):
            resp = client.post(
                "/backup/create",
                json={"kinds": ["snapshot_bundle"], "tier": "daily"},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["tier"] == "daily"
        assert body["success"] is True

    def test_snapshot_bundle_write_error_returns_success_false(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """write_bundle raises BackupError → 200 with success=False."""
        from unittest.mock import patch

        from paramem.backup.types import BackupError

        config = _make_config(tmp_path)
        state = _make_state(tmp_path, config)
        client = _make_client(monkeypatch, state)

        with patch(
            "paramem.backup.backup.write_bundle",
            side_effect=BackupError("episodic slot not found"),
        ):
            resp = client.post("/backup/create", json={"kinds": ["snapshot_bundle"]})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is False
        assert body["error"] is not None


# ---------------------------------------------------------------------------
# _remount_adapters_from_disk — the on-demand re-mount step factored from
# the boot mount machinery
# ---------------------------------------------------------------------------


class TestRemountAdaptersFromDisk:
    def test_cloud_only_is_a_noop(self, tmp_path, monkeypatch) -> None:
        """No resident model (``_state["model"] is None``) -- slots on disk
        are the truth and the next model acquisition mounts them; this call
        must not touch *config* or raise."""
        config = _make_config(tmp_path)
        state = {"model": None, "tokenizer": None}
        monkeypatch.setattr(app_module, "_state", state)

        app_module._remount_adapters_from_disk(config)

        assert state["model"] is None

    def _tier_cfg(self, tmp_path: Path):
        """A real ServerConfig rooted under tmp_path -- tier_config_map()
        needs a real .adapters tree that _make_config (a bare backups-only
        stub) does not carry. target_modules is narrowed to ["q_proj"] on
        every tier so the shape matches _TinyBase (the tree's one minimal
        real-nn.Module PEFT-wrap fixture, tests/_fold_fixtures.py) -- the
        fixture's k_proj/v_proj/o_proj/gate_proj/... targets have no
        matching layer on a model this small."""
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        data_root = tmp_path / "data"
        cfg.paths = PathsConfig(
            data=data_root, sessions=data_root / "sessions", debug=data_root / "debug"
        )
        for tier in ("episodic", "semantic", "procedural"):
            getattr(cfg.adapters, tier).target_modules = ["q_proj"]
        return cfg

    def test_local_mode_detach_then_ensure_resident_tiers_leaves_every_tier_mounted(
        self, tmp_path, monkeypatch
    ) -> None:
        """Local mode: every currently-mounted adapter (including a leftover
        interim family) is detached, then every tier in
        config.tier_config_map() is resident again on the SAME model
        object -- the zero-adapter window between detach and re-create is
        restored. The model-level mount loop (_mount_adapters_from_slots)
        is patched to a no-op here so
        this test isolates the detach -> ensure_resident_tiers sequence
        _remount_adapters_from_disk itself owns; the real mount loop is
        exercised separately below with zero slots on disk."""
        from paramem.models.loader import create_adapter, ensure_resident_tiers
        from tests._fold_fixtures import _TinyBase

        cfg = self._tier_cfg(tmp_path)
        tier_map = cfg.tier_config_map()
        assert tier_map, "precondition: the fixture config must enable at least one tier"

        model = ensure_resident_tiers(_TinyBase(), tier_map)
        create_adapter(model, tier_map["episodic"], "episodic_interim_20260101T0000")

        state = {"model": model, "tokenizer": MagicMock()}
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_mount_adapters_from_slots", lambda *a, **k: None)

        app_module._remount_adapters_from_disk(cfg)

        expected_tiers = set(tier_map)
        assert expected_tiers <= set(model.peft_config)
        assert "episodic_interim_20260101T0000" not in model.peft_config
        assert model.active_adapter in expected_tiers

    def test_zero_mountable_slots_on_disk_still_yields_resident_cold_tiers(
        self, tmp_path, monkeypatch
    ) -> None:
        """A /backup/restore that lands zero adapter weights on disk for any
        tier (e.g. a config-only bundle) still drives the REAL
        _mount_adapters_from_slots without raising, and every configured
        tier ends up resident cold -- no matching slot to mount."""
        from paramem.models.loader import ensure_resident_tiers
        from tests._fold_fixtures import _TinyBase

        cfg = self._tier_cfg(tmp_path)
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)  # empty -- no slot anywhere
        tier_map = cfg.tier_config_map()
        assert tier_map, "precondition: the fixture config must enable at least one tier"

        model = ensure_resident_tiers(_TinyBase(), tier_map)
        state = {"model": model, "tokenizer": MagicMock(), "adapter_manifest_status": {}}
        monkeypatch.setattr(app_module, "_state", state)

        app_module._remount_adapters_from_disk(cfg)

        expected_tiers = set(tier_map)
        assert expected_tiers <= set(model.peft_config)
        assert model.active_adapter in expected_tiers

    def test_boot_and_remount_agree_on_a_divergent_rank_slot(self, tmp_path, monkeypatch) -> None:
        """The identical divergent-rank slot is arbitrated the same way at
        boot (_mount_adapters_from_slots via _load_model_into_state) and at
        /backup/restore's remount (_mount_adapters_from_slots via
        _remount_adapters_from_disk): refused, tier exists-and-cold, the
        SAME manifest row -- never silently adopted on either path."""
        from paramem.adapters.manifest import tier_registry_sha256
        from paramem.adapters.slot import write_slot
        from paramem.models.loader import ensure_resident_tiers, has_prior_trained_weights
        from paramem.training.key_registry import KeyRegistry
        from tests._fold_fixtures import _TinyBase
        from tests._manifest_fixtures import make_train_manifest, write_slot_files

        cfg = self._tier_cfg(tmp_path)
        # The manifest below stamps rank=8 (make_train_manifest's fixed
        # LORA_RANK) -- the live config asks for 16, a divergent rank.
        cfg.adapters.episodic.rank = 16

        tier_root = cfg.adapter_dir / "episodic"
        tier_root.mkdir(parents=True)
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        registry.save(tier_root / "indexed_key_registry.json")
        registry_hash = tier_registry_sha256(tier_root)
        manifest = make_train_manifest(name="episodic", registry_sha256=registry_hash, key_count=1)
        write_slot(
            tier_root,
            manifest=manifest,
            write_payload=lambda pending: write_slot_files(pending),
        )

        tier_map = cfg.tier_config_map()
        assert tier_map, "precondition: the fixture config must enable at least one tier"

        # -- boot: a fresh cold wrap, mounted directly --
        boot_model = ensure_resident_tiers(_TinyBase(), tier_map)
        boot_state: dict = {"adapter_manifest_status": {}}
        app_module._mount_adapters_from_slots(boot_model, MagicMock(), cfg, boot_state)

        # -- remount: a second fresh cold wrap, reached through
        # _remount_adapters_from_disk (the /backup/restore path) --
        remount_model = ensure_resident_tiers(_TinyBase(), tier_map)
        remount_state: dict = {
            "model": remount_model,
            "tokenizer": MagicMock(),
            "adapter_manifest_status": {},
        }
        monkeypatch.setattr(app_module, "_state", remount_state)
        app_module._remount_adapters_from_disk(cfg)

        assert not has_prior_trained_weights(boot_model, "episodic"), (
            "boot must refuse the divergent-rank slot -- episodic stays cold"
        )
        assert not has_prior_trained_weights(remount_model, "episodic"), (
            "remount must refuse the divergent-rank slot identically -- episodic stays cold"
        )

        boot_row = boot_state["adapter_manifest_status"]["episodic"]
        remount_row = remount_state["adapter_manifest_status"]["episodic"]
        assert boot_row["status"] == remount_row["status"] == "mismatch"
        assert boot_row["field"] == remount_row["field"] == "lora.rank"
        assert boot_row["severity"] == remount_row["severity"]
        assert boot_row["slot_path"] == remount_row["slot_path"]


# ---------------------------------------------------------------------------
# _lift_quarantined_store — the ONE re-runnable store-repair primitive
# ---------------------------------------------------------------------------


class TestLiftQuarantinedStore:
    def test_publishes_store_and_rebuilds_router_on_success(self, tmp_path, monkeypatch) -> None:
        config = MagicMock()
        new_store = MagicMock()

        def _fake_preload(cfg, *, model, tokenizer):
            assert cfg is config
            return new_store

        monkeypatch.setattr(app_module, "_preload_memory_store", _fake_preload)

        router_calls = []

        class _FakeRouter:
            def __init__(self, **kwargs):
                router_calls.append(kwargs)

        monkeypatch.setattr(app_module, "QueryRouter", _FakeRouter)

        state = {"model": None, "tokenizer": None, "ha_graph": None, "memory_store": "OLD"}
        monkeypatch.setattr(app_module, "_state", state)

        result = app_module._lift_quarantined_store(config)

        assert result is True
        assert state["memory_store"] is new_store
        assert router_calls and router_calls[0]["memory_store"] is new_store

    def test_returns_false_and_leaves_memory_store_none_when_quarantine_persists(
        self, tmp_path, monkeypatch
    ) -> None:
        config = _make_config(tmp_path)
        monkeypatch.setattr(
            app_module, "_preload_memory_store", lambda cfg, *, model, tokenizer: None
        )

        state = {"model": None, "tokenizer": None, "memory_store": "OLD", "router": "UNCHANGED"}
        monkeypatch.setattr(app_module, "_state", state)

        result = app_module._lift_quarantined_store(config)

        assert result is False
        assert state["memory_store"] is None
        assert state["router"] == "UNCHANGED"
