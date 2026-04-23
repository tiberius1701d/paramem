"""Tests for the 4 backup CLI subcommands (Slice 6b).

Tests cover dispatch, rendering, --json mode, and error handling.

Test numbers reference the plan §9 test matrix:
44 — backup-list dispatch
45 — backup-list renders rows
46 — backup-list --json mode
47 — backup-list --kind filter passthrough
48 — backup-create default kinds
49 — backup-create explicit kinds + label
50 — backup-create renders slots and skips
51 — backup-restore happy path
52 — backup-restore 409 trial → hint
53 — backup-restore 400 restore_kind_not_supported
54 — backup-prune default
55 — backup-prune --dry-run
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
        "restart_required": True,
        "restart_hint": "systemctl --user restart paramem-server",
    }


def _fake_prune_response(dry_run=False):
    return {
        "deleted": ["/data/backups/config/20260301-040000"],
        "preserved_immune": [],
        "preserved_pre_migration_window": ["/data/backups/config/20260420-040000"],
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
# Test 44 — backup-list dispatch
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
# Test 45 — backup-list renders rows
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
# Test 46 — backup-list --json mode
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
# Test 47 — backup-list --kind filter passthrough
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
# Test 48 — backup-create default kinds
# ---------------------------------------------------------------------------


class TestBackupCreateDefaultKinds:
    def test_backup_create_default_kinds_posts_correct_body(self, monkeypatch, capsys) -> None:
        """main(['backup-create']) → POST body kinds=["config","graph","registry"]."""
        seen_bodies = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_create_response()

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-create"])
        assert rc == 0
        assert seen_bodies
        assert seen_bodies[0]["kinds"] == ["config", "graph", "registry"]
        assert seen_bodies[0]["label"] is None


# ---------------------------------------------------------------------------
# Test 49 — backup-create explicit kinds + label
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
# Test 50 — backup-create renders slots and skips
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
# Test 51 — backup-restore happy path
# ---------------------------------------------------------------------------


class TestBackupRestoreHappyPath:
    def test_backup_restore_happy_path(self, monkeypatch, capsys) -> None:
        """Mocked 200 → stdout contains 'Restored config' + restart hint; rc=0."""
        fake_resp = _fake_restore_response("20260421-040000")
        monkeypatch.setattr(http_client, "post_json", lambda *a, **kw: fake_resp)
        args = _args(backup_id="20260421-040000", json=False)
        rc = backup_restore.run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Restored config" in captured.out
        assert "restart" in captured.out.lower() or "systemctl" in captured.out


# ---------------------------------------------------------------------------
# Test 52 — backup-restore 409 trial_active → operator hint
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
# Test 53 — backup-restore 400 restore_kind_not_supported
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
# backup-restore --force-rotate-key plumbing
# ---------------------------------------------------------------------------


class TestBackupRestoreCliPassesForceRotateKeyFlag:
    def test_backup_restore_cli_passes_force_rotate_key_flag(self, monkeypatch) -> None:
        """CLI --force-rotate-key → body includes force_rotate_key=True."""
        seen_bodies: list[dict] = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_restore_response("20260421-040000")

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-restore", "20260421-040000", "--force-rotate-key"])
        assert rc == 0
        assert seen_bodies
        assert seen_bodies[0]["backup_id"] == "20260421-040000"
        assert seen_bodies[0].get("force_rotate_key") is True


class TestBackupRestoreCliDefaultOmitsForceRotateKey:
    def test_backup_restore_cli_default_omits_force_rotate_key(self, monkeypatch) -> None:
        """`paramem backup-restore <id>` → body does NOT include force_rotate_key."""
        seen_bodies: list[dict] = []

        def _fake_post(url, body, **kw):
            seen_bodies.append(body)
            return _fake_restore_response("20260421-040000")

        monkeypatch.setattr(http_client, "post_json", _fake_post)
        rc = main(["backup-restore", "20260421-040000"])
        assert rc == 0
        assert seen_bodies
        assert "force_rotate_key" not in seen_bodies[0]


class TestBackupRestore400FingerprintMismatchSurfacesMessage:
    def test_backup_restore_400_fingerprint_mismatch_surfaces_message(
        self, monkeypatch, capsys
    ) -> None:
        """Server 400 fingerprint_mismatch → stderr carries the force-flag hint; rc=1."""
        import json as _json

        detail = {
            "error": "fingerprint_mismatch",
            "message": (
                "Backup was encrypted with a different key (backup=abc..., current=def...). "
                "Pass force_rotate_key=true to proceed; the prior key must remain set."
            ),
            "backup_fingerprint": "a" * 16,
            "current_fingerprint": "b" * 16,
        }

        def _raise(*a, **kw):
            raise http_client.ServerHTTPError(
                400, "http://x/backup/restore", _json.dumps({"detail": detail})
            )

        monkeypatch.setattr(http_client, "post_json", _raise)
        args = _args(backup_id="20260421-040000", json=False, force_rotate_key=False)
        rc = backup_restore.run(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "force_rotate_key" in captured.err


# ---------------------------------------------------------------------------
# Test 54 — backup-prune default
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
# Test 55 — backup-prune --dry-run
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
