"""Tests for the POST /consolidate/housekeeping endpoint and its dispatch infrastructure.

All tests are CPU-only, no model load required.  The consolidation-loop and
BackgroundTrainer are mocked so the implementation-level dispatch paths can be
verified in isolation.

Coverage:
- ``_consolidation_dispatch_guards`` shared guard helper
- ``_maybe_trigger_housekeeping`` dispatcher (guard pass-through, submission path)
- ``_run_full_consolidation_sync(housekeeping=True)`` gate overrides:
  - B1 fix: ``tiers_rebuilt==[]`` does not trigger the noop early-return
  - Session non-consumption: ``mark_consolidated`` NOT called
  - ``_state["consolidating"]`` cleared on completion
- Route guard: ``POST /consolidate/housekeeping`` requires ``require_admin``
  (introspection test already in test_require_admin.py; this file tests runtime
  behaviour via the dispatch functions, not the HTTP layer, to avoid full-app init)
- Window-stamp preservation: train-mode housekeeping passes the existing stamp
  as ``window_stamp_override`` so ``_is_full_cycle_due`` is not perturbed (B1).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_router() -> MagicMock:
    """Build a minimal mock router that supports reload()."""
    r = MagicMock()
    r.reload.return_value = None
    return r


def _make_hk_state(
    *,
    mode: str = "local",
    consolidating: bool = False,
    bg_is_training: bool = False,
    consolidation_mode: str = "train",
) -> dict:
    """Minimal ``_state`` dict for housekeeping dispatch tests.

    Args:
        mode: Runtime mode ("local" or "cloud-only").
        consolidating: Whether ``_state["consolidating"]`` is already True.
        bg_is_training: Whether the BackgroundTrainer reports active training.
        consolidation_mode: Value for ``config.consolidation.mode``.
    """
    mock_config = MagicMock()
    mock_config.consolidation.mode = consolidation_mode
    # Prevent ThermalPolicy.from_consolidation_config from comparing a MagicMock.
    mock_config.consolidation.training_temp_limit = 0

    mock_loop = MagicMock()
    mock_loop.model = MagicMock(name="model")
    # Default run_housekeeping return: successful noop-ish result with tiers_rebuilt=[].
    mock_loop.run_housekeeping.return_value = {
        "tiers_rebuilt": [],
        "graph_drift_count": 0,
        "drift_deduplicated": 0,
        "drift_orphan": 0,
        "drift_genuine_loss": 0,
        "keys_per_tier": {},
        "recall_per_tier": {},
        "rolled_back": False,
        "rollback_tier": None,
        "tier_delta": {},
    }

    bg = None
    if bg_is_training:
        bg = MagicMock()
        bg.is_training = True

    return {
        "config": mock_config,
        "model": MagicMock(name="model"),
        "tokenizer": MagicMock(name="tokenizer"),
        "consolidation_loop": mock_loop,
        "session_buffer": MagicMock(),
        "router": _make_mock_router(),
        "background_trainer": bg,
        "consolidating": consolidating,
        "mode": mode,
        "cloud_only_reason": None,
        "last_consolidation": None,
        "last_consolidation_result": None,
        "last_consolidation_error": None,
        "event_loop": None,
        "migration": {},
    }


# ---------------------------------------------------------------------------
# TestConsolidationDispatchGuards
# ---------------------------------------------------------------------------


class TestConsolidationDispatchGuards:
    """_consolidation_dispatch_guards returns the right block reason or None."""

    def test_returns_none_when_clear(self, monkeypatch) -> None:
        """All guards pass → returns None (proceed)."""
        import paramem.server.app as app_module

        state = _make_hk_state()
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() is None

    def test_deferred_already_running(self, monkeypatch) -> None:
        """consolidating=True → deferred_already_running."""
        import paramem.server.app as app_module

        state = _make_hk_state(consolidating=True)
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_already_running"

    def test_deferred_cloud_only(self, monkeypatch) -> None:
        """mode=cloud-only → deferred_cloud_only."""
        import paramem.server.app as app_module

        state = _make_hk_state(mode="cloud-only")
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_cloud_only"

    def test_deferred_bg_training(self, monkeypatch) -> None:
        """BackgroundTrainer.is_training=True → deferred_bg_training."""
        import paramem.server.app as app_module

        state = _make_hk_state(bg_is_training=True)
        monkeypatch.setattr(app_module, "_state", state)
        assert app_module._consolidation_dispatch_guards() == "deferred_bg_training"


# ---------------------------------------------------------------------------
# TestMaybeTriggerHousekeeping
# ---------------------------------------------------------------------------


class TestMaybeTriggerHousekeeping:
    """_maybe_trigger_housekeeping gate + submission path."""

    def test_returns_deferred_when_consolidating(self, monkeypatch) -> None:
        """consolidating=True → propagates deferred_already_running."""
        import paramem.server.app as app_module

        state = _make_hk_state(consolidating=True)
        monkeypatch.setattr(app_module, "_state", state)
        result = app_module._maybe_trigger_housekeeping()
        assert result == "deferred_already_running"

    def test_returns_deferred_when_cloud_only(self, monkeypatch) -> None:
        """mode=cloud-only → deferred_cloud_only."""
        import paramem.server.app as app_module

        state = _make_hk_state(mode="cloud-only")
        monkeypatch.setattr(app_module, "_state", state)
        result = app_module._maybe_trigger_housekeeping()
        assert result == "deferred_cloud_only"

    def test_returns_started_housekeeping_and_sets_consolidating(self, monkeypatch) -> None:
        """Clear state → returns started_housekeeping and sets consolidating=True."""

        import paramem.server.app as app_module

        state = _make_hk_state()
        monkeypatch.setattr(app_module, "_state", state)

        _submitted: list[str] = []

        # Patch get_running_loop + run_in_executor so the submission doesn't
        # actually run; capture what was submitted.
        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_loop.run_in_executor.return_value = mock_future
        mock_future.add_done_callback.return_value = None

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            result = app_module._maybe_trigger_housekeeping()

        assert result == "started_housekeeping"
        assert state["consolidating"] is True, (
            "_state['consolidating'] must be set to True before dispatching"
        )
        mock_loop.run_in_executor.assert_called_once()


# ---------------------------------------------------------------------------
# TestRunFullConsolidationSyncHousekeeping
# ---------------------------------------------------------------------------


class TestRunFullConsolidationSyncHousekeeping:
    """_run_full_consolidation_sync(housekeeping=True) gate overrides."""

    def _run_sync(self, state: dict, monkeypatch) -> None:
        """Run _run_full_consolidation_sync(housekeeping=True) with mocked BT."""
        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)
        mock_bt = MagicMock()
        # submit() calls the closure synchronously so we can inspect state after.
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.background_trainer.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync(housekeeping=True)

    def test_b1_fix_tiers_rebuilt_empty_does_not_noop(self, monkeypatch) -> None:
        """B1 fix: housekeeping=True + tiers_rebuilt==[] does NOT trigger noop early-return.

        Under the scheduled fold, an empty tiers_rebuilt signals a no-op and the
        function returns early without calling _save_key_metadata.  Under
        housekeeping=True the same result must be treated as a successful grooming
        pass — _save_key_metadata must still be called.
        """

        state = _make_hk_state()
        # run_housekeeping returns tiers_rebuilt=[] (simulate path, no interims).
        state["consolidation_loop"].run_housekeeping.return_value = {
            "tiers_rebuilt": [],
            "graph_drift_count": 2,
            "drift_deduplicated": 2,
            "drift_orphan": 0,
            "drift_genuine_loss": 0,
            "keys_per_tier": {"episodic": 5},
            "recall_per_tier": {"episodic": 1.0},
            "rolled_back": False,
            "rollback_tier": None,
            "tier_delta": {
                "episodic": {
                    "active_before": 5,
                    "active_after": 5,
                    "staled_by_reason": {},
                    "minted": 0,
                }
            },
        }
        # Disable replay so _finalize_full skips _hydrate_memory_store_in_place.
        state["consolidation_loop"].store.replay_enabled = False

        with (
            patch("paramem.server.consolidation._save_key_metadata") as mock_save_meta,
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        (
            mock_save_meta.assert_called_once(),
            ("B1 fix: _save_key_metadata must be called even when tiers_rebuilt==[]"),
        )

    def test_sessions_not_marked_consolidated(self, monkeypatch) -> None:
        """housekeeping=True: session_buffer.mark_consolidated is NOT called.

        The housekeeping fold re-grooms the existing knowledge but does not
        consume pending sessions — they stay pending for the next scheduled tick.
        """

        state = _make_hk_state()
        # Return a successful result with non-empty tiers_rebuilt.
        state["consolidation_loop"].run_housekeeping.return_value = {
            "tiers_rebuilt": ["episodic"],
            "graph_drift_count": 0,
            "drift_deduplicated": 0,
            "drift_orphan": 0,
            "drift_genuine_loss": 0,
            "keys_per_tier": {"episodic": 10},
            "recall_per_tier": {"episodic": 1.0},
            "rolled_back": False,
            "rollback_tier": None,
            "tier_delta": {},
        }
        state["consolidation_loop"].store.replay_enabled = False

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        (
            state["session_buffer"].mark_consolidated.assert_not_called(),
            ("housekeeping=True: mark_consolidated must NOT be called"),
        )

    def test_consolidating_cleared_on_completion(self, monkeypatch) -> None:
        """_state['consolidating'] is cleared (set to False) after the fold completes."""

        state = _make_hk_state()
        state["consolidating"] = True  # set by dispatcher before submit
        state["consolidation_loop"].run_housekeeping.return_value = {
            "tiers_rebuilt": [],
            "graph_drift_count": 0,
            "drift_deduplicated": 0,
            "drift_orphan": 0,
            "drift_genuine_loss": 0,
            "keys_per_tier": {},
            "recall_per_tier": {},
            "rolled_back": False,
            "rollback_tier": None,
            "tier_delta": {},
        }
        state["consolidation_loop"].store.replay_enabled = False

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        assert state["consolidating"] is False, (
            "_state['consolidating'] must be cleared after fold completes"
        )

    def test_run_housekeeping_called_not_consolidate_interim_adapters(self, monkeypatch) -> None:
        """_run_full_consolidation_sync(housekeeping=True) calls run_housekeeping, not
        consolidate_interim_adapters or consolidate_interim_graphs directly."""

        state = _make_hk_state(consolidation_mode="train")
        state["consolidation_loop"].store.replay_enabled = False

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._revalidate_main_adapter_manifests"),
        ):
            self._run_sync(state, monkeypatch)

        mock_loop = state["consolidation_loop"]
        mock_loop.run_housekeeping.assert_called_once()
        mock_loop.consolidate_interim_adapters.assert_not_called()
        mock_loop.consolidate_interim_graphs.assert_not_called()


# ---------------------------------------------------------------------------
# TestWindowStampPreservation (B1 regression guard)
# ---------------------------------------------------------------------------


def _make_train_loop_with_stamp(output_dir: Path, window_stamp: str) -> "object":
    """Build a minimal ConsolidationLoop in train mode, seeded with a known window stamp.

    Writes ``window_stamp`` to the lex-max main episodic slot's ``meta.json`` so
    ``_last_full_consolidation_window(output_dir)`` returns it.

    Returns the loop with ``consolidate_interim_adapters`` replaced by a MagicMock
    so no GPU or real training is required.
    """
    from paramem.training.consolidation import ConsolidationLoop

    # Write the known stamp into a fake main episodic slot's meta.json.
    slot_dir = output_dir / "episodic" / "20260101T000000Z"
    slot_dir.mkdir(parents=True, exist_ok=True)
    (slot_dir / "meta.json").write_text(
        json.dumps({"window_stamp": window_stamp, "schema_version": 2}),
        encoding="utf-8",
    )

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.output_dir = output_dir

    # config.mode="train" so run_housekeeping takes the train branch.
    cfg = MagicMock()
    cfg.mode = "train"
    loop.config = cfg

    # Replace consolidate_interim_adapters with a mock that returns a minimal result.
    loop.consolidate_interim_adapters = MagicMock(
        return_value={
            "tiers_rebuilt": ["episodic"],
            "graph_drift_count": 0,
            "drift_deduplicated": 0,
            "drift_orphan": 0,
            "drift_genuine_loss": 0,
            "keys_per_tier": {"episodic": 1},
            "recall_per_tier": {"episodic": 1.0},
            "rolled_back": False,
            "rollback_tier": None,
            "tier_delta": {},
        }
    )
    return loop


class TestWindowStampPreservation:
    """B1 regression: train-mode housekeeping must not advance the window stamp.

    ``run_housekeeping`` reads the existing stamp from the lex-max main slot's
    ``meta.json`` and passes it as ``window_stamp_override`` to
    ``consolidate_interim_adapters``.  Without this, ``_save_adapters`` floors the
    stamp to the current period on every housekeeping run, which perturbs
    ``_is_full_cycle_due``'s identity check and can trigger a duplicate scheduled
    cycle immediately after a housekeeping run.
    """

    def test_window_stamp_override_passed_to_consolidate(self, tmp_path: Path) -> None:
        """run_housekeeping (train mode) passes the existing window stamp as
        window_stamp_override to consolidate_interim_adapters.

        Seeds a known stamp in the main episodic slot's meta.json, calls
        run_housekeeping, and asserts that consolidate_interim_adapters received
        ``window_stamp_override=<seed>`` — proving the stamp is preserved, not
        recomputed from the current period.
        """
        known_stamp = "202601010000"  # a fixed stamp that would not be today's floor
        loop = _make_train_loop_with_stamp(tmp_path, known_stamp)

        loop.run_housekeeping()

        loop.consolidate_interim_adapters.assert_called_once()
        _call_kwargs = loop.consolidate_interim_adapters.call_args
        assert _call_kwargs is not None, "consolidate_interim_adapters was not called"
        # Extract keyword arguments from the call.
        _, kwargs = _call_kwargs
        assert "window_stamp_override" in kwargs, (
            "run_housekeeping must pass window_stamp_override to consolidate_interim_adapters; "
            f"actual kwargs: {kwargs}"
        )
        assert kwargs["window_stamp_override"] == known_stamp, (
            f"window_stamp_override must equal the stamp read from meta.json ({known_stamp!r}); "
            f"got {kwargs['window_stamp_override']!r}"
        )

    def test_housekeeping_true_passed_to_consolidate(self, tmp_path: Path) -> None:
        """run_housekeeping (train mode) passes housekeeping=True to consolidate_interim_adapters.

        Sanity-check: the housekeeping=True flag that bypasses the accumulate gate
        must be forwarded regardless of stamp handling.
        """
        loop = _make_train_loop_with_stamp(tmp_path, "202601010000")

        loop.run_housekeeping()

        _, kwargs = loop.consolidate_interim_adapters.call_args
        assert kwargs.get("housekeeping") is True, (
            f"run_housekeeping must forward housekeeping=True; actual kwargs: {kwargs}"
        )

    def test_no_main_slot_passes_none_as_override(self, tmp_path: Path) -> None:
        """run_housekeeping (train mode) with no main slot passes window_stamp_override=None.

        When there is no lex-max main episodic slot (fresh install), the stamp
        is None and None is forwarded to consolidate_interim_adapters.  Under
        window_stamp_override=None, _save_adapters floors the stamp to the current
        period (the correct behavior for a fresh full cycle).
        """
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path  # no episodic/ subdirectory

        cfg = MagicMock()
        cfg.mode = "train"
        loop.config = cfg

        loop.consolidate_interim_adapters = MagicMock(
            return_value={
                "tiers_rebuilt": [],
                "graph_drift_count": 0,
                "drift_deduplicated": 0,
                "drift_orphan": 0,
                "drift_genuine_loss": 0,
                "keys_per_tier": {},
                "recall_per_tier": {},
                "rolled_back": False,
                "rollback_tier": None,
                "tier_delta": {},
            }
        )

        loop.run_housekeeping()

        _, kwargs = loop.consolidate_interim_adapters.call_args
        assert "window_stamp_override" in kwargs, (
            "window_stamp_override must be passed even when there is no main slot; "
            f"actual kwargs: {kwargs}"
        )
        assert kwargs["window_stamp_override"] is None, (
            "window_stamp_override must be None when there is no main slot; "
            f"got {kwargs['window_stamp_override']!r}"
        )
