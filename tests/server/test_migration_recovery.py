"""Tests for paramem.server.migration_recovery.

Covers all 5 crash-recovery cases plus edge cases (unparseable marker,
multiple orphan backups, max_age_hours default).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paramem.backup.backup import write as backup_write
from paramem.backup.types import ArtifactKind
from paramem.server.migration_recovery import (
    RecoveryAction,
    recover_migration_state,
)
from paramem.server.trial_state import (
    TRIAL_MARKER_SCHEMA_VERSION,
    TrialMarker,
    write_trial_marker,
)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_config(path: Path, content: bytes = b"model: mistral\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_pre_migration_backup(
    backups_root: Path,
    pre_trial_hash: str,
    data: bytes = b"config: live\n",
) -> Path:
    """Write a pre_migration config backup slot and return its path."""
    return backup_write(
        ArtifactKind.CONFIG,
        data,
        meta_fields={"tier": "pre_migration", "pre_trial_hash": pre_trial_hash},
        base_dir=backups_root / "config",
    )


def _make_marker(
    tmp_path: Path,
    pre_trial_config_sha256: str,
    candidate_config_sha256: str = "c" * 64,
) -> TrialMarker:
    return TrialMarker(
        schema_version=TRIAL_MARKER_SCHEMA_VERSION,
        started_at="2026-04-22T01:00:00+00:00",
        pre_trial_config_sha256=pre_trial_config_sha256,
        candidate_config_sha256=candidate_config_sha256,
        backup_paths={
            "config": str(tmp_path / "backups" / "config" / "20260422-010000"),
        },
        trial_adapter_dir=str(tmp_path / "state" / "trial_adapter"),
        trial_graph_dir=str(tmp_path / "state" / "trial_graph"),
        config_artifact_filename="config-20260422-010000.bin",
    )


# ---------------------------------------------------------------------------
# Case 1: RESUME_TRIAL
# ---------------------------------------------------------------------------


class TestCase1ResumeTrial:
    def test_recovery_case_1_resume_trial(self, tmp_path):
        """trial.json exists; live_config_hash != marker.pre_trial_config_sha256 → RESUME_TRIAL."""
        live_content = b"model: mistral\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", live_content)
        # Marker records a DIFFERENT pre-trial hash (swap already happened).
        old_hash = "a" * 64
        state_dir = tmp_path / "data/ha/state"
        marker = _make_marker(tmp_path, pre_trial_config_sha256=old_hash)
        write_trial_marker(state_dir, marker)

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )
        assert result.action == RecoveryAction.RESUME_TRIAL
        assert result.trial_marker is not None
        assert result.trial_marker.pre_trial_config_sha256 == old_hash
        assert result.recovery_required == []


# ---------------------------------------------------------------------------
# Case 2: STEP3_CRASH_CLEANUP
# ---------------------------------------------------------------------------


class TestCase2Step3Crash:
    def test_recovery_case_2_step3_crash(self, tmp_path):
        """trial.json present AND hashes equal → STEP3_CRASH_CLEANUP; marker deleted."""
        live_content = b"model: mistral\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", live_content)
        live_hash = _sha256(live_content)
        state_dir = tmp_path / "data/ha/state"
        # Marker records the SAME hash as live config → rename never happened.
        marker = _make_marker(tmp_path, pre_trial_config_sha256=live_hash)
        write_trial_marker(state_dir, marker)

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )
        assert result.action == RecoveryAction.STEP3_CRASH_CLEANUP
        assert result.trial_marker is None
        # Marker must be deleted.
        from paramem.server.trial_state import read_trial_marker

        assert read_trial_marker(state_dir) is None
        # Log should warn.
        levels = [lvl for lvl, _ in result.log_lines]
        assert "WARNING" in levels


# ---------------------------------------------------------------------------
# Case 3: ORPHAN_SWEEP
# ---------------------------------------------------------------------------


class TestCase3OrphanSweep:
    def test_recovery_case_3_orphan_sweep_in_window(self, tmp_path):
        """No marker; pre_migration backup; pre_trial_hash == sha256(live); young → ORPHAN_SWEEP."""
        live_content = b"model: mistral\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", live_content)
        live_hash = _sha256(live_content)
        backups_root = tmp_path / "data/ha/backups"
        slot = _write_pre_migration_backup(backups_root, pre_trial_hash=live_hash)
        state_dir = tmp_path / "data/ha/state"

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=backups_root,
            max_age_hours=24,
        )
        assert result.action == RecoveryAction.ORPHAN_SWEEP
        # The slot should be deleted.
        assert not slot.exists()
        levels = [lvl for lvl, _ in result.log_lines]
        assert "WARNING" in levels

    def test_recovery_case_3_orphan_sweep_outside_window(self, tmp_path, monkeypatch):
        """Same as above but backup is outside the window → NORMAL_LIVE (orphan left in place)."""
        live_content = b"model: mistral\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", live_content)
        live_hash = _sha256(live_content)
        backups_root = tmp_path / "data/ha/backups"
        slot = _write_pre_migration_backup(backups_root, pre_trial_hash=live_hash)
        state_dir = tmp_path / "data/ha/state"

        # Monkeypatch datetime.now inside migration_recovery to return a time
        # 48h AFTER the backup was created → backup is outside the 24h window.
        import paramem.server.migration_recovery as _mr

        real_now = datetime.now(tz=timezone.utc)
        far_future = real_now + timedelta(hours=48)

        monkeypatch.setattr(
            _mr,
            "datetime",
            type(
                "FakeDatetime",
                (),
                {
                    "now": staticmethod(lambda tz=None: far_future),
                    "utcnow": staticmethod(lambda: far_future.replace(tzinfo=None)),
                },
            ),
        )

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=backups_root,
            max_age_hours=24,
        )
        # Orphan is outside window → treated as normal live.
        assert result.action == RecoveryAction.NORMAL_LIVE
        # Slot must NOT be deleted.
        assert slot.exists()


# ---------------------------------------------------------------------------
# Case 4: AMBIGUOUS_REQUIRES_OPERATOR
# ---------------------------------------------------------------------------


class TestCase4Ambiguous:
    def test_recovery_case_4_ambiguous(self, tmp_path):
        """No marker; pre_migration backup with pre_trial_hash != sha256(live) → AMBIGUOUS."""
        live_content = b"model: mistral\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", live_content)
        backups_root = tmp_path / "data/ha/backups"
        # Different pre_trial_hash (some other SHA256).
        slot = _write_pre_migration_backup(backups_root, pre_trial_hash="d" * 64)
        state_dir = tmp_path / "data/ha/state"

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=backups_root,
        )
        assert result.action == RecoveryAction.AMBIGUOUS_REQUIRES_OPERATOR
        assert len(result.recovery_required) > 0
        assert "RECOVERY REQUIRED" in result.recovery_required[0]
        levels = [lvl for lvl, _ in result.log_lines]
        assert "ERROR" in levels
        # AMBIGUOUS leaves the backup in place — operator must inspect it.
        assert slot.exists(), "AMBIGUOUS recovery must NOT delete the orphan backup slot"


# ---------------------------------------------------------------------------
# Case 5: NORMAL_LIVE
# ---------------------------------------------------------------------------


class TestCase5NormalLive:
    def test_recovery_case_5_normal_live(self, tmp_path):
        """No marker, no backups → NORMAL_LIVE."""
        live_config = _write_config(tmp_path / "configs/server.yaml")
        result = recover_migration_state(
            state_dir=tmp_path / "data/ha/state",
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )
        assert result.action == RecoveryAction.NORMAL_LIVE
        assert result.recovery_required == []
        assert result.log_lines == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_recovery_unparseable_marker_treated_as_ambiguous(self, tmp_path):
        """trial.json with garbage JSON → AMBIGUOUS; marker NOT deleted."""
        live_config = _write_config(tmp_path / "configs/server.yaml")
        state_dir = tmp_path / "data/ha/state"
        state_dir.mkdir(parents=True)
        (state_dir / "trial.json").write_text("NOT JSON", encoding="utf-8")

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )
        assert result.action == RecoveryAction.AMBIGUOUS_REQUIRES_OPERATOR
        # Marker must NOT be deleted.
        assert (state_dir / "trial.json").exists()
        levels = [lvl for lvl, _ in result.log_lines]
        assert "ERROR" in levels

    def test_recovery_max_age_hours_default_24(self, tmp_path):
        """No explicit max_age_hours → default 24h used; recent backup swept."""
        live_content = b"model: mistral\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", live_content)
        live_hash = _sha256(live_content)
        backups_root = tmp_path / "data/ha/backups"
        slot = _write_pre_migration_backup(backups_root, pre_trial_hash=live_hash)
        state_dir = tmp_path / "data/ha/state"

        # No max_age_hours argument → defaults to 24h.
        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=backups_root,
        )
        assert result.action == RecoveryAction.ORPHAN_SWEEP
        assert not slot.exists()

    def test_recovery_missing_live_config(self, tmp_path):
        """live_config_path does not exist → case 5 (no marker, no backup)."""
        result = recover_migration_state(
            state_dir=tmp_path / "state",
            live_config_path=tmp_path / "nonexistent.yaml",
            backups_root=tmp_path / "backups",
        )
        assert result.action == RecoveryAction.NORMAL_LIVE


# ---------------------------------------------------------------------------
# Base-swap recovery: RESUME_BASE_SWAP
# ---------------------------------------------------------------------------


def _make_base_swap_marker(
    tmp_path: Path,
    base_swap_phase: str,
    live_hash: str,
) -> TrialMarker:
    """Build a base-swap TrialMarker for recovery tests.

    Parameters
    ----------
    tmp_path:
        Pytest tmp directory.
    base_swap_phase:
        One of ``"phaseA"``, ``"phaseA_done"``, ``"phaseB"``, ``"done"``.
    live_hash:
        The hash stored in ``pre_trial_config_sha256`` (used as the Mistral
        config hash captured before Phase A renamed the config).
    """
    bundle_slot = tmp_path / "backups" / "bundles" / "20260524-000000"
    bundle_slot.mkdir(parents=True, exist_ok=True)
    return TrialMarker(
        schema_version=TRIAL_MARKER_SCHEMA_VERSION,
        started_at="2026-05-24T00:00:00+00:00",
        pre_trial_config_sha256=live_hash,
        candidate_config_sha256="c" * 64,
        backup_paths={"bundle": str(bundle_slot)},
        trial_adapter_dir=str(tmp_path / "state" / "trial_adapter"),
        trial_graph_dir=str(tmp_path / "state" / "trial_graph"),
        config_artifact_filename="",
        migration_kind="base_swap",
        base_swap_phase=base_swap_phase,
        old_model="mistral",
        new_model="qwen3-4b",
        bundle_slot=str(bundle_slot),
    )


class TestBaseSwapRecovery:
    """Base-swap markers yield RESUME_BASE_SWAP, not RESUME_TRIAL (disambiguation first)."""

    def test_phaseA_done_marker_returns_resume_base_swap(self, tmp_path):
        """A base-swap marker at phaseA_done → RESUME_BASE_SWAP, not RESUME_TRIAL.

        After Phase A the live config was renamed (live_hash != pre_trial_hash),
        which is exactly the condition that would normally trigger RESUME_TRIAL.
        The migration_kind check must come FIRST so this is classified correctly.
        """
        # Phase A renames live config → live_config now has the candidate hash.
        candidate_content = b"model: qwen3-4b\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", candidate_content)
        # pre_trial_hash recorded the OLD (Mistral) config hash — different from live.
        mistral_hash = "a" * 64
        state_dir = tmp_path / "data/ha/state"
        marker = _make_base_swap_marker(tmp_path, "phaseA_done", mistral_hash)
        write_trial_marker(state_dir, marker)

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )

        assert result.action == RecoveryAction.RESUME_BASE_SWAP, (
            f"Expected RESUME_BASE_SWAP, got {result.action!r}. "
            "migration_kind='base_swap' must be checked before hash comparison."
        )
        assert result.trial_marker is not None
        assert result.trial_marker.migration_kind == "base_swap"
        assert result.trial_marker.base_swap_phase == "phaseA_done"
        assert result.recovery_required == []

    def test_phaseB_marker_returns_resume_base_swap(self, tmp_path):
        """A base-swap marker at phaseB → RESUME_BASE_SWAP."""
        candidate_content = b"model: qwen3-4b\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", candidate_content)
        mistral_hash = "b" * 64
        state_dir = tmp_path / "data/ha/state"
        marker = _make_base_swap_marker(tmp_path, "phaseB", mistral_hash)
        write_trial_marker(state_dir, marker)

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )

        assert result.action == RecoveryAction.RESUME_BASE_SWAP
        assert result.trial_marker is not None
        assert result.trial_marker.base_swap_phase == "phaseB"

    def test_base_swap_marker_not_classified_as_resume_trial(self, tmp_path):
        """migration_kind='base_swap' is NEVER classified as RESUME_TRIAL.

        This is the critical disambiguation: a base-swap marker looks identical
        to a normal trial RESUME_TRIAL (live_hash != pre_trial_hash) but must
        take the base-swap recovery path, not the trial path.
        """
        candidate_content = b"model: qwen3-4b\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", candidate_content)
        mistral_hash = "a" * 64  # different from live → would trigger RESUME_TRIAL without guard
        state_dir = tmp_path / "data/ha/state"

        # Try both mid-flight phases.
        for phase in ("phaseA_done", "phaseB"):
            marker = _make_base_swap_marker(tmp_path, phase, mistral_hash)
            write_trial_marker(state_dir, marker)

            result = recover_migration_state(
                state_dir=state_dir,
                live_config_path=live_config,
                backups_root=tmp_path / "data/ha/backups",
            )

            assert result.action != RecoveryAction.RESUME_TRIAL, (
                f"phase={phase!r}: base-swap marker must not be classified as RESUME_TRIAL"
            )
            assert result.action == RecoveryAction.RESUME_BASE_SWAP

    def test_base_swap_marker_carries_trial_marker(self, tmp_path):
        """RESUME_BASE_SWAP result carries the parsed marker in trial_marker."""
        candidate_content = b"model: qwen3-4b\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", candidate_content)
        state_dir = tmp_path / "data/ha/state"
        marker = _make_base_swap_marker(tmp_path, "phaseA_done", "a" * 64)
        write_trial_marker(state_dir, marker)

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )

        assert result.action == RecoveryAction.RESUME_BASE_SWAP
        assert result.trial_marker is not None
        assert result.trial_marker.old_model == "mistral"
        assert result.trial_marker.new_model == "qwen3-4b"

    def test_base_swap_result_has_no_recovery_required(self, tmp_path):
        """RESUME_BASE_SWAP is a normal resume path; recovery_required is empty."""
        candidate_content = b"model: qwen3-4b\n"
        live_config = _write_config(tmp_path / "configs/server.yaml", candidate_content)
        state_dir = tmp_path / "data/ha/state"
        marker = _make_base_swap_marker(tmp_path, "phaseB", "b" * 64)
        write_trial_marker(state_dir, marker)

        result = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config,
            backups_root=tmp_path / "data/ha/backups",
        )

        assert result.action == RecoveryAction.RESUME_BASE_SWAP
        assert result.recovery_required == []
        # INFO line emitted but no ERROR.
        levels = {lvl for lvl, _ in result.log_lines}
        assert "ERROR" not in levels
        assert "INFO" in levels
