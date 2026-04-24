"""Unit tests for paramem.server.attention (Slice 5a).

Tests cover every populator's emit/no-emit branches, collection ordering,
stub behaviour, and the AttentionItem dataclass contract.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from paramem.server.attention import (
    AttentionItem,
    _age_seconds_from_iso,
    _collect_adapter_fingerprint_items,
    _collect_backup_items,
    _collect_config_drift_items,
    _collect_consolidation_items,
    _collect_encryption_items,
    _collect_key_rotation_items,
    _collect_migration_items,
    _collect_pre_flight_items,
    _collect_sweeper_items,
    collect_attention_items,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _live_state(**overrides) -> dict:
    """Minimal state dict for a clean LIVE server."""
    state: dict = {
        "migration": {
            "state": "LIVE",
            "recovery_required": [],
            "shape_changes": [],
            "trial": {},
            "staged_at": None,
        },
        "config_drift": {"detected": False, "loaded_hash": "abc", "disk_hash": "abc"},
        "consolidating": False,
        "session_buffer": None,
        "adapter_manifest_status": {},
    }
    state.update(overrides)
    return state


def _trial_state(gates_status: str = "pending", pending_count: int = 0) -> dict:
    """Minimal state dict for a TRIAL server."""
    buf = MagicMock()
    buf.pending_count = pending_count
    return _live_state(
        migration={
            "state": "TRIAL",
            "recovery_required": [],
            "shape_changes": [],
            "staged_at": None,
            "trial": {
                "started_at": "2026-04-22T00:00:00+00:00",
                "gates": {
                    "status": gates_status,
                    "details": [],
                    "completed_at": "2026-04-22T00:10:00+00:00",
                },
            },
        },
        session_buffer=buf,
    )


# ---------------------------------------------------------------------------
# AttentionItem dataclass contract
# ---------------------------------------------------------------------------


def test_attention_item_kind_is_str_not_literal():
    """kind field annotation must be str (not Literal) — forward-compat rule."""
    fields = {f.name: f for f in dataclasses.fields(AttentionItem)}
    # The type hint is str; Literal types would show as typing.Literal[...].
    assert fields["kind"].type is str or fields["kind"].type == "str"


def test_attention_item_frozen():
    """AttentionItem is frozen — mutation raises FrozenInstanceError."""
    item = AttentionItem(kind="test", level="info", summary="s", action_hint=None, age_seconds=None)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        item.kind = "other"  # type: ignore[misc]


def test_attention_item_to_dict():
    """to_dict() returns all 5 fields."""
    item = AttentionItem(
        kind="migration_trial_pass",
        level="action_required",
        summary="TRIAL active",
        action_hint="paramem migrate-accept",
        age_seconds=600,
    )
    d = item.to_dict()
    assert d == {
        "kind": "migration_trial_pass",
        "level": "action_required",
        "summary": "TRIAL active",
        "action_hint": "paramem migrate-accept",
        "age_seconds": 600,
    }


# ---------------------------------------------------------------------------
# _age_seconds_from_iso
# ---------------------------------------------------------------------------


def test_age_seconds_handles_naive_iso():
    """Naive ISO string (no TZ) is treated as UTC — no exception."""
    age = _age_seconds_from_iso("2000-01-01T00:00:00")
    assert isinstance(age, int)
    assert age > 0


def test_age_seconds_handles_unparseable():
    """Garbage input → None, no exception."""
    assert _age_seconds_from_iso("garbage string") is None


def test_age_seconds_handles_empty():
    """Empty string → None."""
    assert _age_seconds_from_iso("") is None


def test_age_seconds_handles_none():
    """None input → None."""
    assert _age_seconds_from_iso(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _collect_migration_items
# ---------------------------------------------------------------------------


def test_migration_live_no_recovery_returns_empty():
    """LIVE state with no recovery_required → no items."""
    state = _live_state()
    items = _collect_migration_items(state)
    assert items == []


def test_migration_live_with_recovery_emits_info_per_string():
    """recovery_required=["A","B"] → 2 items, kind=migration_recovery_required."""
    state = _live_state()
    state["migration"]["recovery_required"] = ["msg A", "msg B"]
    items = _collect_migration_items(state)
    assert len(items) == 2
    assert all(it.kind == "migration_recovery_required" for it in items)
    assert all(it.level == "info" for it in items)
    assert items[0].summary == "msg A"
    assert items[1].summary == "msg B"


def test_migration_staging_with_shape_changes():
    """STAGING + shape_changes → 1 item, kind=migration_shape_change_pending."""
    state = _live_state()
    state["migration"]["state"] = "STAGING"
    state["migration"]["shape_changes"] = [
        {
            "adapter": "episodic",
            "field": "rank",
            "old_value": 8,
            "new_value": 16,
            "consequence": "prior weights discarded",
        }
    ]
    state["migration"]["staged_at"] = "2026-04-22T00:00:00+00:00"
    items = _collect_migration_items(state)
    assert len(items) == 1
    assert items[0].kind == "migration_shape_change_pending"
    assert items[0].level == "info"
    assert items[0].action_hint is not None


def test_migration_staging_no_shape_changes():
    """STAGING + shape_changes=[] → 0 items."""
    state = _live_state()
    state["migration"]["state"] = "STAGING"
    state["migration"]["shape_changes"] = []
    items = _collect_migration_items(state)
    assert items == []


def test_migration_staging_truncates_shape_changes():
    """STAGING + 5 shape changes → summary contains '+2 more' truncation suffix."""
    state = _live_state()
    state["migration"]["state"] = "STAGING"
    state["migration"]["shape_changes"] = [
        {"adapter": f"adapter_{i}", "field": "rank", "old_value": 8, "new_value": 16}
        for i in range(5)
    ]
    state["migration"]["staged_at"] = "2026-04-22T00:00:00+00:00"
    items = _collect_migration_items(state)
    assert len(items) == 1
    assert items[0].kind == "migration_shape_change_pending"
    # Only first 3 are shown; 5 - 3 = 2 overflow.
    assert "+2 more" in items[0].summary


def test_migration_staging_exactly_three_shape_changes_no_truncation():
    """STAGING + exactly 3 shape changes → no '+N more' suffix in summary."""
    state = _live_state()
    state["migration"]["state"] = "STAGING"
    state["migration"]["shape_changes"] = [
        {"adapter": f"adapter_{i}", "field": "rank", "old_value": 8, "new_value": 16}
        for i in range(3)
    ]
    state["migration"]["staged_at"] = "2026-04-22T00:00:00+00:00"
    items = _collect_migration_items(state)
    assert len(items) == 1
    assert "more" not in items[0].summary


def test_migration_trial_pending_emits_running():
    """TRIAL + gates.status=pending → 1 item trial_running, level=info."""
    state = _trial_state("pending")
    items = _collect_migration_items(state)
    running = [it for it in items if it.kind == "migration_trial_running"]
    assert len(running) == 1
    assert running[0].level == "info"
    assert running[0].action_hint is None


def test_migration_trial_pass_emits_action_required():
    """TRIAL + gates pass → 1 item trial_pass, level=action_required."""
    state = _trial_state("pass")
    items = _collect_migration_items(state)
    pass_items = [it for it in items if it.kind == "migration_trial_pass"]
    assert len(pass_items) == 1
    assert pass_items[0].level == "action_required"
    assert pass_items[0].action_hint is not None
    assert "accept" in pass_items[0].action_hint.lower()


def test_migration_trial_no_new_sessions_eligible():
    """TRIAL + gates no_new_sessions → 1 item kind=migration_trial_pass (accept-eligible)."""
    state = _trial_state("no_new_sessions")
    items = _collect_migration_items(state)
    assert any(it.kind == "migration_trial_pass" for it in items)


def test_migration_trial_failed_emits_failed():
    """TRIAL + gates fail → 1 item trial_failed, level=failed."""
    state = _trial_state("fail")
    state["migration"]["trial"]["gates"]["details"] = [
        {
            "gate": 1,
            "name": "extraction",
            "status": "fail",
            "reason": "extraction crashed",
            "metrics": None,
        }
    ]
    items = _collect_migration_items(state)
    failed = [it for it in items if it.kind == "migration_trial_failed"]
    assert len(failed) == 1
    assert failed[0].level == "failed"
    assert "FAILED" in failed[0].summary
    assert failed[0].action_hint == "paramem migrate-rollback"


def test_migration_trial_exception_emits_failed():
    """TRIAL + gates trial_exception → 1 item trial_failed, level=failed."""
    state = _trial_state("trial_exception")
    state["migration"]["trial"]["gates"]["exception"] = "model not loaded"
    items = _collect_migration_items(state)
    failed = [it for it in items if it.kind == "migration_trial_failed"]
    assert len(failed) == 1
    assert "model not loaded" in failed[0].summary


# ---------------------------------------------------------------------------
# _collect_consolidation_items (Fix 1)
# ---------------------------------------------------------------------------


def test_consolidation_blocked_during_trial_pending():
    """migration_state=TRIAL + gates.status=pending → 1 item consolidation_blocked."""
    state = _trial_state("pending", pending_count=3)
    items = _collect_consolidation_items(state)
    assert len(items) == 1
    assert items[0].kind == "consolidation_blocked"
    assert "3" in items[0].summary


def test_consolidation_blocked_during_trial_none_status():
    """migration_state=TRIAL + gates=None (just kicked off) → 1 item."""
    buf = MagicMock()
    buf.pending_count = 5
    state = _live_state(
        migration={
            "state": "TRIAL",
            "recovery_required": [],
            "shape_changes": [],
            "staged_at": None,
            "trial": {
                "started_at": "2026-04-22T00:00:00+00:00",
                "gates": None,  # trial just kicked off
            },
        },
        session_buffer=buf,
    )
    items = _collect_consolidation_items(state)
    assert len(items) == 1
    assert "5" in items[0].summary


def test_consolidation_normal_run_no_item():
    """consolidating=True + migration_state=LIVE → 0 items (Fix 1 verified)."""
    state = _live_state(consolidating=True)
    items = _collect_consolidation_items(state)
    assert items == []


def test_consolidation_flag_true_trial_pass_no_item():
    """consolidating=True + TRIAL gates=pass → 0 items (gates finished, not pending)."""
    state = _trial_state("pass")
    state["consolidating"] = True
    items = _collect_consolidation_items(state)
    assert items == []


def test_consolidation_blocked_no_session_buffer():
    """TRIAL pending + no session_buffer → queued shown as 'unknown'."""
    buf = MagicMock()
    buf.pending_count = 0
    state = _trial_state("pending")
    state["session_buffer"] = None  # explicitly None
    items = _collect_consolidation_items(state)
    assert len(items) == 1
    assert "unknown" in items[0].summary


# ---------------------------------------------------------------------------
# _collect_sweeper_items
# ---------------------------------------------------------------------------


def test_sweeper_held_when_pending_in_trial():
    """TRIAL + buffer.pending_count=3 → 1 item sweeper_held."""
    state = _trial_state("pending", pending_count=3)
    items = _collect_sweeper_items(state)
    assert len(items) == 1
    assert items[0].kind == "sweeper_held"
    assert "3 transcripts" in items[0].summary


def test_sweeper_no_item_in_live():
    """LIVE + pending=3 → 0 items (sweeper not blocked outside TRIAL)."""
    buf = MagicMock()
    buf.pending_count = 3
    state = _live_state(session_buffer=buf)
    items = _collect_sweeper_items(state)
    assert items == []


def test_sweeper_no_item_when_no_pending():
    """TRIAL + pending_count=0 → 0 items."""
    state = _trial_state("pending", pending_count=0)
    items = _collect_sweeper_items(state)
    assert items == []


# ---------------------------------------------------------------------------
# _collect_config_drift_items
# ---------------------------------------------------------------------------


def test_config_drift_detected_emits():
    """config_drift.detected=True → 1 item, kind=config_drift, action_required."""
    state = _live_state(
        config_drift={
            "detected": True,
            "loaded_hash": "aaa",
            "disk_hash": "bbb",
            "last_checked_at": "2026-04-22T01:00:00+00:00",
        }
    )
    items = _collect_config_drift_items(state)
    assert len(items) == 1
    assert items[0].kind == "config_drift"
    assert items[0].level == "action_required"
    assert items[0].action_hint is not None


def test_config_drift_clean_no_item():
    """config_drift.detected=False → 0 items."""
    state = _live_state(config_drift={"detected": False})
    items = _collect_config_drift_items(state)
    assert items == []


def test_config_drift_missing_key_no_item():
    """Missing config_drift key → 0 items (no exception)."""
    state = _live_state()
    del state["config_drift"]
    items = _collect_config_drift_items(state)
    assert items == []


# ---------------------------------------------------------------------------
# _collect_adapter_fingerprint_items
# ---------------------------------------------------------------------------


def test_adapter_fingerprint_primary_emits_red():
    """episodic adapter with severity=red + status=mismatch → 1 item, level=failed."""
    state = _live_state(
        adapter_manifest_status={
            "episodic": {
                "status": "mismatch",
                "reason": "base_model.sha mismatch",
                "field": "base_model.sha",
                "severity": "red",
                "slot_path": "/adapters/episodic",
                "checked_at": "2026-04-22T00:00:00+00:00",
            }
        }
    )
    items = _collect_adapter_fingerprint_items(state)
    assert len(items) == 1
    assert items[0].kind == "adapter_fingerprint_mismatch_primary"
    assert items[0].level == "failed"
    assert "episodic" in items[0].summary
    assert "DISABLED" in items[0].summary


def test_adapter_fingerprint_secondary_emits_info():
    """semantic adapter with severity=yellow → 1 item, kind=secondary, level=info."""
    state = _live_state(
        adapter_manifest_status={
            "semantic": {
                "status": "mismatch",
                "reason": "tokenizer.sha mismatch",
                "field": "tokenizer.sha",
                "severity": "yellow",
                "slot_path": "/adapters/semantic",
                "checked_at": "2026-04-22T00:00:00+00:00",
            }
        }
    )
    items = _collect_adapter_fingerprint_items(state)
    assert len(items) == 1
    assert items[0].kind == "adapter_fingerprint_mismatch_secondary"
    assert items[0].level == "info"
    assert "semantic" in items[0].summary


def test_adapter_fingerprint_primary_before_secondary():
    """Both rows present → primary item appears before secondary."""
    state = _live_state(
        adapter_manifest_status={
            "episodic": {
                "status": "mismatch",
                "reason": "sha mismatch",
                "field": "base_model.sha",
                "severity": "red",
                "slot_path": "/adapters/episodic",
                "checked_at": "",
            },
            "semantic": {
                "status": "mismatch",
                "reason": "vocab mismatch",
                "field": "tokenizer.sha",
                "severity": "yellow",
                "slot_path": "/adapters/semantic",
                "checked_at": "",
            },
        }
    )
    items = _collect_adapter_fingerprint_items(state)
    assert len(items) == 2
    assert items[0].kind == "adapter_fingerprint_mismatch_primary"
    assert items[1].kind == "adapter_fingerprint_mismatch_secondary"


def test_adapter_fingerprint_ok_no_item():
    """status=ok → no item."""
    state = _live_state(
        adapter_manifest_status={
            "episodic": {
                "status": "ok",
                "severity": "green",
                "reason": None,
                "field": None,
                "slot_path": "/a",
                "checked_at": "",
            }
        }
    )
    items = _collect_adapter_fingerprint_items(state)
    assert items == []


def test_adapter_fingerprint_manifest_missing():
    """status=manifest_missing → item emitted (missing manifest is mismatch)."""
    state = _live_state(
        adapter_manifest_status={
            "procedural": {
                "status": "manifest_missing",
                "reason": "no meta.json found",
                "field": None,
                "severity": "yellow",
                "slot_path": "/adapters/procedural",
                "checked_at": "",
            }
        }
    )
    items = _collect_adapter_fingerprint_items(state)
    assert len(items) == 1
    assert items[0].kind == "adapter_fingerprint_mismatch_secondary"


# ---------------------------------------------------------------------------
# Stub populators
# ---------------------------------------------------------------------------


def test_stub_populators_return_empty():
    """Stub populators return [] when config=None (forward-compat guardrail)."""
    state = _live_state()
    # _collect_backup_items and _collect_pre_flight_items now take (state, config) after Slice 6b.
    # Passing config=None exercises the early-return guard.
    assert _collect_backup_items(state, None) == []
    assert _collect_key_rotation_items(state) == []
    assert _collect_pre_flight_items(state, None) == []


# ---------------------------------------------------------------------------
# _collect_encryption_items
# ---------------------------------------------------------------------------


def test_encryption_items_empty_when_posture_on():
    """Security: ON → no encryption attention item."""
    state = _live_state(encryption="on")
    assert _collect_encryption_items(state) == []


def test_encryption_items_empty_when_posture_absent():
    """Pre-lifespan / test shim with no encryption field → no item (not a crash)."""
    state = _live_state()
    assert "encryption" not in state
    assert _collect_encryption_items(state) == []


def test_encryption_items_fires_on_security_off():
    """Security: OFF → one action_required item pointing at generate-key."""
    state = _live_state(encryption="off")
    items = _collect_encryption_items(state)

    assert len(items) == 1
    item = items[0]
    assert item.kind == "encryption_off"
    assert item.level == "action_required"
    assert "SECURITY: OFF" in item.summary
    assert item.action_hint is not None
    assert "generate-key" in item.action_hint
    assert "PARAMEM_DAILY_PASSPHRASE" in item.action_hint


# ---------------------------------------------------------------------------
# collect_attention_items — ordering and integration
# ---------------------------------------------------------------------------


def test_collect_order_matches_spec():
    """Items from all 5 active populators appear in spec order."""
    buf = MagicMock()
    buf.pending_count = 2
    state = {
        "migration": {
            "state": "TRIAL",
            "recovery_required": [],
            "shape_changes": [],
            "staged_at": None,
            "trial": {
                "started_at": "2026-04-22T00:00:00+00:00",
                "gates": {"status": "pending", "details": [], "completed_at": None},
            },
        },
        "config_drift": {
            "detected": True,
            "loaded_hash": "abc",
            "disk_hash": "def",
            "last_checked_at": "2026-04-22T00:00:00+00:00",
        },
        "consolidating": False,
        "session_buffer": buf,
        "adapter_manifest_status": {
            "episodic": {
                "status": "mismatch",
                "reason": "sha mismatch",
                "field": "base_model.sha",
                "severity": "red",
                "slot_path": "/adapters/episodic",
                "checked_at": "",
            }
        },
    }
    items = collect_attention_items(state, None)
    kinds = [it.kind for it in items]
    # Expected order: migration → consolidation → sweeper → config_drift → adapter_fingerprint
    migration_idx = kinds.index("migration_trial_running")
    consolidation_idx = kinds.index("consolidation_blocked")
    sweeper_idx = kinds.index("sweeper_held")
    config_idx = kinds.index("config_drift")
    adapter_idx = kinds.index("adapter_fingerprint_mismatch_primary")
    assert migration_idx < consolidation_idx < sweeper_idx < config_idx < adapter_idx


def test_collect_empty_when_live_clean():
    """LIVE server with no alerts → collect_attention_items returns []."""
    state = _live_state()
    items = collect_attention_items(state, None)
    assert items == []
