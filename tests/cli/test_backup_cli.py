"""Tests for the 4 backup CLI subcommands.

Tests cover dispatch, rendering, --json mode, and error handling for
backup-list, backup-create, backup-restore, and backup-prune.
"""

from __future__ import annotations

import json

from paramem.cli import backup_create, backup_list, backup_restore, http_client
from paramem.cli.main import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_list_response(items=None, disk_used=0, disk_cap=0):
    return {
        "items": items or [],
        "disk_used_bytes": disk_used,
        "disk_cap_bytes": disk_cap,
    }


def _fake_create_response(
    success=True,
    tier="manual",
    written_slots=None,
    skipped=None,
    error=None,
):
    return {
        "success": success,
        "tier": tier,
        "written_slots": written_slots or {"config": "/data/backups/config/20260421/"},
        "skipped_artifacts": skipped or [],
        "error": error,
    }


def _fake_restore_response(backup_id="20260421-040000"):
    return {
        "restored": {"config": "/configs/server.yaml"},
        "backed_up_pre_restore": {"config": "/data/backups/config/safety/"},
        "restored_adapters": [],
        "pruned_orphans": [],
        "serving": True,
        "quarantine_cause": None,
    }


def _fake_prune_response(dry_run=False):
    return {
        "deleted": ["/data/backups/config/20260301-040000"],
        "preserved_immune": [],
        "preserved_migration_window": ["/data/backups/config/20260420-040000"],
        "would_delete_next": [],
        "disk_usage_before": {
            "total_bytes": 3_500_000_000,
            "by_tier": {"daily": 3_500_000_000},
            "cap_bytes": 20_000_000_000,
            "pct_of_cap": 0.175,
        },
        "disk_usage_after": {
            "total_bytes": 1_200_000_000,
            "by_tier": {"daily": 1_200_000_000},
            "cap_bytes": 20_000_000_000,
            "pct_of_cap": 0.060,
        },
        "invalid_slots": [],
        "dry_run": dry_run,
    }


def _args(server_url="http://127.0.0.1:8420", **kwargs):
    """Build a minimal argparse-like namespace."""
    import argparse

    ns = argparse.Namespace(server_url=server_url)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# backup-list dispatch
# ---------------------------------------------------------------------------


class TestBackupListDispatch:
    def test_backup_list_dispatch_calls_backup_list_run(self, monkeypatch, capsys) -> None:
        """main(['backup-list']) dispatches to backup_list.run."""
        called_with = []

        def _fake_run(args):
            called_with.append(args)
            return 0

        monkeypatch.setattr(backup_list, "run", _fake_run)
        rc = main(["backup-list"])
        assert rc == 0
        assert len(called_with) == 1


# ---------------------------------------------------------------------------
# backup-list renders rows
# ---------------------------------------------------------------------------


class TestBackupListRendersRows:
    def test_backup_list_renders_rows(self, monkeypatch, capsys) -> None:
        """Mocked response → stdout has header + one row per item."""
        items = [
            {
                "backup_id": "20260421-040000",
                "kind": "config",
                "tier": "daily",
                "timestamp": "2026-04-21T04:00:00+00:00",
                "size_bytes": 12300,
                "label": None,
                "path": "/data/backups/config/20260421-040000",
            }
        ]
        monkeypatch.setattr(
            http_client,
            "get_json",
            lambda *a, **kw: _fake_list_response(items=items, disk_used=12300),
        )
        args = _args(kind=None, json=False)
        rc = backup_list.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "20260421-040000" in captured.out
        assert "config" in captured.out
        assert "daily" in captured.out


# ---------------------------------------------------------------------------
# backup-list --json mode
# ---------------------------------------------------------------------------


class TestBackupListJsonMode:
    def test_backup_list_json_mode(self, monkeypatch, capsys) -> None:
        """--json → stdout is parsable JSON echoing the response."""
        fake_resp = _fake_list_response(disk_used=1000)
        monkeypatch.setattr(http_client, "get_json", lambda *a, **kw: fake_resp)
        args = _args(kind=None, json=True)
        rc = backup_list.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["disk_used_bytes"] == 1000


# ---------------------------------------------------------------------------
# backup-list --kind filter passthrough
# ---------------------------------------------------------------------------


class TestBackupListKindFilterPassthrough:
    def test_backup_list_kind_config_request_url_contains_kind_param(
        self, monkeypatch, capsys
    ) -> None:
        """--kind config → request URL contains ?kind=config."""
        seen_urls = []

        def _fake_get(url, **kw):
            seen_urls.append(url)
            return _fake_list_response()

        monkeypatch.setattr(http_client, "get_json", _fake_get)
        args = _args(kind="config", json=False)
        backup_list.run(args)
        assert seen_urls, "get_json must have been called"
        assert "kind=config" in seen_urls[0], f"URL {seen_urls[0]!r} missing kind=config"


# ---------------------------------------------------------------------------
# backup-create default kinds
# ---------------------------------------------------------------------------


class TestBackupCreateDefaultKinds:
    def test_backup_create_default_kinds_posts_correct_body(self, monkeypatch, capsys) -> None:
        """main(['backup-create']) with no --kinds → body omits kinds/label entirely.

        The server's BackupCreateRequest default (["snapshot_bundle"]) is the
        single source of truth; the CLI must not hardcode a stale default.
        """
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_create_response()

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-create"])
        assert rc == 0
        assert seen_bodies
        assert "kinds" not in seen_bodies[0], (
            f"kinds must be omitted so the server default applies: {seen_bodies[0]!r}"
        )
        assert "label" not in seen_bodies[0], (
            f"label must be omitted when not given: {seen_bodies[0]!r}"
        )


# ---------------------------------------------------------------------------
# backup-create explicit kinds + label
# ---------------------------------------------------------------------------


class TestBackupCreateExplicitKindsLabel:
    def test_backup_create_explicit_kinds_label(self, monkeypatch, capsys) -> None:
        """--kinds config,registry --label x → correct POST body."""
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_create_response(written_slots={"config": "/c/", "registry": "/r/"})

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-create", "--kinds", "config,registry", "--label", "x"])
        assert rc == 0
        assert seen_bodies
        assert seen_bodies[0]["kinds"] == ["config", "registry"]
        assert seen_bodies[0]["label"] == "x"


# ---------------------------------------------------------------------------
# backup-create renders slots and skips
# ---------------------------------------------------------------------------


class TestBackupCreateRendersSlotsAndSkips:
    def test_backup_create_renders_slots_and_skips(self, monkeypatch, capsys) -> None:
        """Mocked response with 1 skip → stdout lists 1 written + 1 skipped."""
        fake_resp = _fake_create_response(
            written_slots={"config": "/data/backups/config/20260421/"},
            skipped=[{"kind": "registry", "reason": "registry empty (no keys yet)"}],
        )
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(kinds="config,registry", label=None, json=False)
        rc = backup_create.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "config" in captured.out
        assert "registry" in captured.out
        assert "skipped" in captured.out.lower() or "empty" in captured.out.lower()


# ---------------------------------------------------------------------------
# backup-restore happy path
# ---------------------------------------------------------------------------


class TestBackupRestoreHappyPath:
    def test_backup_restore_happy_path(self, monkeypatch, capsys) -> None:
        """Mocked 200 with serving=True → stdout contains 'Restored backup' and
        reports the server is serving with no restart needed; rc=0."""
        fake_resp = _fake_restore_response("20260421-040000")
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(backup_id="20260421-040000", restore_config=False, json=False)
        rc = backup_restore.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Restored backup" in captured.out
        assert "no restart needed" in captured.out.lower()

    def test_backup_restore_default_sends_restore_config_false(self, monkeypatch) -> None:
        """No --restore-config → POST body has restore_config=False (always present)."""
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_restore_response("20260421-040000")

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        args = _args(backup_id="20260421-040000", restore_config=False, json=False)
        backup_restore.run(args)
        assert seen_bodies
        assert seen_bodies[0]["restore_config"] is False

    def test_backup_restore_restore_config_flag_sends_true(self, monkeypatch) -> None:
        """--restore-config → POST body has restore_config=True."""
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_restore_response("20260421-040000")

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        args = _args(backup_id="20260421-040000", restore_config=True, json=False)
        backup_restore.run(args)
        assert seen_bodies
        assert seen_bodies[0]["restore_config"] is True

    def test_backup_restore_snapshot_bundle_renders_adapters_and_orphans(
        self, monkeypatch, capsys
    ) -> None:
        """A snapshot_bundle restore response renders restored_adapters/pruned_orphans

        and does NOT print the old config-only 'Restored config' header.
        """
        fake_resp = {
            "restored": {"registry": "/data/backups/registry/latest"},
            "backed_up_pre_restore": {"bundle": "/data/backups/snapshot/safety"},
            "serving": True,
            "quarantine_cause": None,
            "restored_adapters": ["episodic", "semantic"],
            "pruned_orphans": [
                {"name": "procedural_interim_3", "kind": "interim", "active_keys": 12}
            ],
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(backup_id="20260421-040000", restore_config=True, json=False)
        rc = backup_restore.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Restored config" not in captured.out
        assert "episodic" in captured.out
        assert "semantic" in captured.out
        assert "procedural_interim_3" in captured.out
        assert "interim" in captured.out


# ---------------------------------------------------------------------------
# backup-restore non-serving outcomes — restart advice vs. offline-store advice
# ---------------------------------------------------------------------------


class TestBackupRestoreNonServingOutcomes:
    def test_config_kind_restore_advises_restart(self, monkeypatch, capsys) -> None:
        """A config-kind restore always reports serving=False with no
        quarantine_cause -- the renderer must recognise this shape
        (backed_up_pre_restore keyed 'config') and advise a restart rather
        than misreading it as an offline store."""
        fake_resp = {
            "restored": {"config": "/configs/server.yaml"},
            "backed_up_pre_restore": {"config": "/data/backups/config/safety/"},
            "restored_adapters": [],
            "pruned_orphans": [],
            "serving": False,
            "quarantine_cause": None,
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(backup_id="20260421-040000", restore_config=False, json=False)
        rc = backup_restore.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "restart the server" in captured.out.lower()

    def test_bundle_restore_config_true_advises_restart(self, monkeypatch, capsys) -> None:
        """A snapshot_bundle restore with restore_config=True leaves the
        store on its existing restart posture (the base model may have
        changed) -- the renderer must advise a restart, not read the
        absent quarantine_cause as a healthy converge."""
        fake_resp = {
            "restored": {
                "episodic": "/data/adapters/episodic/slot",
                "config": "/configs/server.yaml",
            },
            "backed_up_pre_restore": {"bundle": "/data/backups/snapshot/safety"},
            "restored_adapters": ["episodic"],
            "pruned_orphans": [],
            "serving": False,
            "quarantine_cause": None,
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(backup_id="20260421-040000", restore_config=True, json=False)
        rc = backup_restore.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "restart the server" in captured.out.lower()

    def test_bundle_restore_quarantined_advises_erase_or_restore(self, monkeypatch, capsys) -> None:
        """A snapshot_bundle restore whose post-restore lift re-quarantined
        the store must name the cause and both ways out -- erase the
        affected keys, or restore a healthy backup -- never the removed
        'Retry the restore' framing."""
        fake_resp = {
            "restored": {"episodic": "/data/adapters/episodic/slot"},
            "backed_up_pre_restore": {"bundle": "/data/backups/snapshot/safety"},
            "restored_adapters": ["episodic"],
            "pruned_orphans": [],
            "serving": False,
            "quarantine_cause": {"message": "episodic (torn_train_slot)"},
        }
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(backup_id="20260421-040000", restore_config=False, json=False)
        rc = backup_restore.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "episodic (torn_train_slot)" in captured.out
        assert "/debug/erase-keys" in captured.out
        assert "restore a healthy backup" in captured.out
        assert "retry the restore" not in captured.out.lower()


# ---------------------------------------------------------------------------
# backup-restore 409 trial_active → operator hint
# ---------------------------------------------------------------------------


class TestBackupRestore409TrialPrintsHint:
    def test_backup_restore_409_trial_prints_hint(self, monkeypatch, capsys) -> None:
        """Mocked 409 trial_active → stderr mentions migrate-accept/rollback; rc=1."""
        detail = {"error": "trial_active", "state": "TRIAL", "message": "..."}
        import json as _json

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                409, "http://x/backup/restore", _json.dumps({"detail": detail})
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        args = _args(backup_id="20260421-040000", json=False)
        rc = backup_restore.run(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "migrate-accept" in captured.err or "migrate-rollback" in captured.err, (
            f"Expected migrate-accept/rollback hint in: {captured.err!r}"
        )


# ---------------------------------------------------------------------------
# backup-restore 400 restore_kind_not_supported
# ---------------------------------------------------------------------------


class TestBackupRestore400KindRejects:
    def test_backup_restore_400_kind_rejects(self, monkeypatch, capsys) -> None:
        """Mocked 400 restore_kind_not_supported → stderr includes server message; rc=1."""
        detail = {
            "error": "restore_kind_not_supported",
            "message": "Only kind='config' restore is supported.",
        }
        import json as _json

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                400, "http://x/backup/restore", _json.dumps({"detail": detail})
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        args = _args(backup_id="20260421-040000", json=False)
        rc = backup_restore.run(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "config" in captured.err.lower() or "kind" in captured.err.lower(), (
            f"Expected kind/config in stderr: {captured.err!r}"
        )


# ---------------------------------------------------------------------------
# backup-prune default
# ---------------------------------------------------------------------------


class TestBackupPruneDefault:
    def test_backup_prune_default(self, monkeypatch, capsys) -> None:
        """main(['backup-prune']) → POST body dry_run=False; stdout shows before/after."""
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_prune_response(dry_run=False)

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-prune"])
        assert rc == 0
        assert seen_bodies
        assert seen_bodies[0]["dry_run"] is False
        captured = capsys.readouterr()
        assert "before" in captured.out.lower()
        assert "after" in captured.out.lower()


# ---------------------------------------------------------------------------
# backup-prune --dry-run
# ---------------------------------------------------------------------------


class TestBackupPruneDryRun:
    def test_backup_prune_dry_run(self, monkeypatch, capsys) -> None:
        """--dry-run → POST body dry_run=True; stdout has 'would delete'."""
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_prune_response(dry_run=True)

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-prune", "--dry-run"])
        assert rc == 0
        assert seen_bodies
        assert seen_bodies[0]["dry_run"] is True
        captured = capsys.readouterr()
        assert "would delete" in captured.out.lower()
