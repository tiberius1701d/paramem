"""Integration tests for app lifespan scheduling and debounce gates.

Tests cover:
1. ConsolidationScheduleConfig.training_idle_debounce_s field validation.
2. _dispatch_consolidation idle-debounce gate.

All GPU/model calls are mocked — no hardware required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# TestIdleDebounceConfig — ConsolidationScheduleConfig.training_idle_debounce_s
# ---------------------------------------------------------------------------


class TestIdleDebounceConfig:
    """ConsolidationScheduleConfig.training_idle_debounce_s field validation."""

    def test_debounce_default_30_seconds(self) -> None:
        """training_idle_debounce_s defaults to 30."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.training_idle_debounce_s == 30

    def test_debounce_negative_rejected(self) -> None:
        """Negative training_idle_debounce_s raises ValueError."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="training_idle_debounce_s must be >= 0"):
            ConsolidationScheduleConfig(training_idle_debounce_s=-1)

    def test_debounce_zero_allowed(self) -> None:
        """training_idle_debounce_s=0 is valid (disables the gate)."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(training_idle_debounce_s=0)
        assert cfg.training_idle_debounce_s == 0


# ---------------------------------------------------------------------------
# TestAbortQuiesceTimeoutConfig — ConsolidationScheduleConfig.abort_quiesce_timeout_s
# ---------------------------------------------------------------------------


class TestAbortQuiesceTimeoutConfig:
    """ConsolidationScheduleConfig.abort_quiesce_timeout_s field validation."""

    def test_default_30_seconds(self) -> None:
        """abort_quiesce_timeout_s defaults to 30.0."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.abort_quiesce_timeout_s == 30.0

    def test_zero_rejected(self) -> None:
        """abort_quiesce_timeout_s=0.0 raises ValueError."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="abort_quiesce_timeout_s must be > 0"):
            ConsolidationScheduleConfig(abort_quiesce_timeout_s=0.0)

    def test_negative_rejected(self) -> None:
        """Negative abort_quiesce_timeout_s raises ValueError."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="abort_quiesce_timeout_s must be > 0"):
            ConsolidationScheduleConfig(abort_quiesce_timeout_s=-1.0)


# ---------------------------------------------------------------------------
# TestSchedulerIdleDebounce — _dispatch_consolidation gate
# ---------------------------------------------------------------------------


def _make_scheduler_state(last_chat_monotonic=None, debounce_s: int = 30) -> tuple:
    """Return (state_patch_dict, config_mock) for scheduler debounce tests.

    These tests focus on the idle-debounce gate only.  ``_is_full_cycle_due``
    is patched to ``False`` in each caller that needs to reach the pending-
    session path — callers add that patch to their ``with`` block.
    """
    cfg = MagicMock()
    cfg.consolidation.training_idle_debounce_s = debounce_s
    # Calendar-exact cadence — keeps the suspend/power-off catch-up gate
    # (systemd_timer.heartbeat_seconds) a no-op so these tests exercise only
    # the idle-debounce gate.
    cfg.consolidation.refresh_cadence = "12h"

    buf = MagicMock()
    buf.pending_facts.return_value = []

    state_patch = {
        "consolidating": False,
        "mode": "local",
        "background_trainer": None,
        "config": cfg,
        "session_buffer": buf,
        "speaker_store": None,
        "pending_rehydration": False,
        "store_load_degraded": False,
        "last_chat_monotonic": last_chat_monotonic,
    }
    return state_patch, cfg


class TestSchedulerIdleDebounce:
    """_dispatch_consolidation returns 'deferred_idle' within the window."""

    def test_scheduled_tick_returns_deferred_idle_within_debounce_window(self) -> None:
        """Tick arriving 5 s after /chat with debounce=30 returns 'deferred_idle'."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(
            last_chat_monotonic=time.monotonic() - 5,
            debounce_s=30,
        )
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result == "deferred_idle", (
            f"Expected 'deferred_idle' within debounce window but got {result!r}"
        )

    def test_scheduled_tick_proceeds_after_debounce_elapsed(self) -> None:
        """Tick arriving 60 s after /chat with debounce=30 proceeds past the gate."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(
            last_chat_monotonic=time.monotonic() - 60,
            debounce_s=30,
        )
        # The tick should reach the no-pending check and return noop_no_pending
        # (session_buffer.pending_facts() returns [] from the mock).
        # _is_full_cycle_due is patched False so this test exercises the debounce
        # gate in isolation without triggering the full-cycle event-loop path.
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result != "deferred_idle", (
            f"Expected tick to proceed past debounce gate but got {result!r}"
        )

    def test_scheduled_tick_debounce_zero_disables_gate(self) -> None:
        """debounce_s=0 disables the gate even when chat fired right now."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(
            last_chat_monotonic=time.monotonic(),
            debounce_s=0,
        )
        # _is_full_cycle_due is patched False so this test exercises the debounce
        # gate in isolation without triggering the full-cycle event-loop path.
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result != "deferred_idle", f"debounce_s=0 must disable gate; got {result!r}"

    def test_scheduled_tick_no_chat_yet_proceeds(self) -> None:
        """last_chat_monotonic=None skips the gate (no /chat has fired yet)."""
        import paramem.server.app as app_module

        state_patch, _ = _make_scheduler_state(last_chat_monotonic=None, debounce_s=30)
        # _is_full_cycle_due is patched False so this test exercises the debounce
        # gate in isolation without triggering the full-cycle event-loop path.
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO, apply_schedule_gate=True
            )

        assert result != "deferred_idle", f"No chat yet must not defer; got {result!r}"
