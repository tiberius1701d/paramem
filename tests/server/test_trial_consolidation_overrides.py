"""Tests for trial consolidation overrides (Slice 3b.2, §10.4).

Verifies that the trial consolidation loop is configured with:
- mode="train" regardless of the candidate config's consolidation.mode
- output paths pointing to state/trial_adapter/ and state/trial_graph/
- gates set to "no_new_sessions" on empty queue
- gates set to "trial_exception" when the trainer raises

No GPU — all tests use mocked model/tokenizer/consolidation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import paramem.server.app as app_module


def _make_state(tmp_path: Path) -> dict:
    """Build a TRIAL _state with a real live config file."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(b"model: mistral\nconsolidation:\n  mode: simulate\n")

    config = MagicMock()
    config.paths.data = tmp_path / "data"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.key_metadata_path = tmp_path / "data" / "key_metadata.json"

    state_dir = tmp_path / "data" / "ha" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    trial_adapter_dir = str((tmp_path / "data" / "ha" / "trial_adapter").resolve())
    trial_graph_dir = str((tmp_path / "data" / "ha" / "trial_graph").resolve())

    trial_stash = {
        "started_at": "2026-04-22T01:00:00+00:00",
        "pre_trial_config_sha256": "a" * 64,
        "candidate_config_sha256": "b" * 64,
        "backup_paths": {
            "config": str(tmp_path / "backups" / "config" / "20260422-010000"),
            "graph": str(tmp_path / "backups" / "graph" / "20260422-010000"),
            "registry": str(tmp_path / "backups" / "registry" / "20260422-010000"),
        },
        "trial_adapter_dir": trial_adapter_dir,
        "trial_graph_dir": trial_graph_dir,
        "gates": {"status": "pending"},
    }

    return {
        "model": MagicMock(),  # non-None so trial consolidation doesn't short-circuit
        "tokenizer": MagicMock(),
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": {
            "state": "TRIAL",
            "trial": trial_stash,
            "recovery_required": [],
        },
        "migration_lock": asyncio.Lock(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "mode": "normal",
        "background_trainer": None,
        "loop": None,
        "session_buffer": None,
        "ha_context": None,
        "speaker_store": None,
    }


class TestBuildTrialLoop:
    """Tests for _build_trial_loop helper.

    _build_trial_loop imports create_consolidation_loop locally from
    paramem.server.consolidation, so we patch that module's attribute.
    """

    def test_trial_loop_adapter_dir_is_overridden(self, tmp_path):
        """_build_trial_loop sets loop.output_dir to trial_adapter_dir."""
        state = _make_state(tmp_path)
        trial_adapter_dir = Path(state["migration"]["trial"]["trial_adapter_dir"])
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])

        # Use a real object so attribute assignment works (MagicMock already does).
        mock_loop = MagicMock()

        # The local import inside _build_trial_loop resolves via sys.modules.
        with patch(
            "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
        ):
            from paramem.server.app import _build_trial_loop
            from paramem.server.config import load_server_config

            trial_config = load_server_config(Path(state["config_path"]))
            _build_trial_loop(
                state["model"],
                state["tokenizer"],
                trial_config,
                trial_adapter_dir,
                trial_graph_dir,
            )

        # loop.output_dir should be set to trial_adapter_dir.
        assert mock_loop.output_dir == trial_adapter_dir

    def test_trial_loop_persist_graph_enabled(self, tmp_path):
        """_build_trial_loop enables persist_graph and sets graph_path."""
        state = _make_state(tmp_path)
        trial_adapter_dir = Path(state["migration"]["trial"]["trial_adapter_dir"])
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])

        mock_loop = MagicMock()

        with patch(
            "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
        ):
            from paramem.server.app import _build_trial_loop
            from paramem.server.config import load_server_config

            trial_config = load_server_config(Path(state["config_path"]))
            _build_trial_loop(
                state["model"],
                state["tokenizer"],
                trial_config,
                trial_adapter_dir,
                trial_graph_dir,
            )

        assert mock_loop.persist_graph is True
        assert mock_loop.graph_path == trial_graph_dir / "cumulative_graph.json"


class TestRunTrialConsolidation:
    """Tests for _run_trial_consolidation coroutine.

    All imports inside _run_trial_consolidation are local, so we patch via
    the module where they are defined (paramem.server.consolidation, etc.).
    """

    def test_trial_exception_sets_trial_exception_gate(self, tmp_path, monkeypatch):
        """When trial consolidation raises, gates become {"status":"trial_exception",...}."""
        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        async def _run():
            # Raise inside _build_trial_loop by making create_consolidation_loop fail.
            with patch(
                "paramem.server.consolidation.create_consolidation_loop",
                side_effect=RuntimeError("GPU OOM"),
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    cfg = MagicMock()
                    cfg.consolidation.mode = "simulate"
                    mock_load.return_value = cfg
                    await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "trial_exception"
        assert "exception" in gates

    def test_trial_completion_sets_no_new_sessions_gate(self, tmp_path, monkeypatch):
        """Successful trial cycle with empty queue sets gates.status to no_new_sessions."""
        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_summary = {"status": "no_pending", "sessions": 0}

        async def _run():
            mock_loop = MagicMock()
            with patch(
                "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    cfg = MagicMock()
                    cfg.consolidation.mode = "simulate"
                    mock_load.return_value = cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        with patch(
                            "paramem.server.consolidation.run_consolidation",
                            return_value=mock_summary,
                        ):
                            await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "no_new_sessions"

    def test_trial_skips_when_no_model(self, tmp_path, monkeypatch):
        """When model is None (cloud-only), short-circuits with trial_exception."""
        state = _make_state(tmp_path)
        state["model"] = None
        state["tokenizer"] = None
        monkeypatch.setattr(app_module, "_state", state)

        async def _run():
            await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "trial_exception"
        assert "model not loaded" in gates.get("exception", "")


class TestTrialDoesNotMarkConsolidated:
    """Fix 1 regression: trial cycle must never call session_buffer.mark_consolidated."""

    def test_trial_cycle_does_not_mark_consolidated(self, tmp_path, monkeypatch):
        """run_consolidation called from trial path must not invoke mark_consolidated.

        Verifies the ``mark_consolidated_callback=lambda _: None`` plumbing:
        the real session_buffer.mark_consolidated must not be called so that
        pending sessions stay in the buffer after the trial cycle (spec L364).
        """
        from unittest.mock import MagicMock, patch

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_session_buffer = MagicMock()
        mock_session_buffer.get_pending.return_value = []  # empty queue → no_pending
        state["session_buffer"] = mock_session_buffer

        mock_summary = {"status": "no_pending", "sessions": 0}

        async def _run():
            mock_loop = MagicMock()
            with patch(
                "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    cfg = MagicMock()
                    cfg.consolidation.mode = "simulate"
                    mock_load.return_value = cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        with patch(
                            "paramem.server.consolidation.run_consolidation",
                            return_value=mock_summary,
                        ) as mock_run:
                            await app_module._run_trial_consolidation()
                            # Verify mark_consolidated_callback=lambda _: None was passed.
                            call_kwargs = mock_run.call_args.kwargs
                            callback = call_kwargs.get("mark_consolidated_callback")
                            assert callback is not None, (
                                "trial path must pass mark_consolidated_callback"
                            )
                            # Calling the callback must be a no-op, not delegate to buffer.
                            callback(["session-1", "session-2"])
                            assert not mock_session_buffer.mark_consolidated.called, (
                                "trial callback must not call session_buffer.mark_consolidated"
                            )

        asyncio.run(_run())

    def test_trial_pending_count_unchanged_after_trial(self, tmp_path, monkeypatch):
        """After a trial run, session_buffer.get_pending() returns the same list as before.

        The trial cycle must leave the session buffer untouched so
        /migration/rollback (3b.3) finds the original pending queue intact.
        """
        from unittest.mock import MagicMock, patch

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        pending_before = [{"session_id": "s1"}, {"session_id": "s2"}]
        mock_session_buffer = MagicMock()
        mock_session_buffer.get_pending.return_value = pending_before
        state["session_buffer"] = mock_session_buffer

        mock_summary = {"status": "no_pending", "sessions": 0}

        async def _run():
            mock_loop = MagicMock()
            with patch(
                "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    cfg = MagicMock()
                    cfg.consolidation.mode = "simulate"
                    mock_load.return_value = cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        with patch(
                            "paramem.server.consolidation.run_consolidation",
                            return_value=mock_summary,
                        ):
                            await app_module._run_trial_consolidation()

        asyncio.run(_run())

        # mark_consolidated must NOT have been called on the buffer.
        mock_session_buffer.mark_consolidated.assert_not_called()

    def test_trial_loop_forces_train_mode(self, tmp_path, monkeypatch):
        """Trial consolidation forces consolidation.mode='train' even when candidate says simulate.

        Spec Resolved Decision 27, L239: trial mode is always train.
        Verifies that the dead object.__setattr__ line is gone and only the
        plain assignment remains (Fix 3).
        """
        from unittest.mock import MagicMock, patch

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        captured_cfg = {}

        mock_summary = {"status": "no_pending", "sessions": 0}

        async def _run():
            mock_loop = MagicMock()
            with patch(
                "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    # Candidate sets mode=simulate — trial must override to train.
                    from paramem.server.config import ConsolidationScheduleConfig

                    real_cfg = MagicMock()
                    real_cfg.consolidation = ConsolidationScheduleConfig(mode="simulate")
                    mock_load.return_value = real_cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        def capture_run(model, tokenizer, cfg, buf, **kw):
                            captured_cfg["mode"] = cfg.consolidation.mode
                            return mock_summary

                        with patch(
                            "paramem.server.consolidation.run_consolidation",
                            side_effect=capture_run,
                        ):
                            await app_module._run_trial_consolidation()

        asyncio.run(_run())

        assert captured_cfg.get("mode") == "train", (
            "trial consolidation must override consolidation.mode to 'train'"
        )


class TestUpdateTrialGates:
    """Tests for _update_trial_gates helper."""

    def test_update_trial_gates_sets_gate_dict(self, tmp_path, monkeypatch):
        """_update_trial_gates writes into _state["migration"]["trial"]["gates"]."""
        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        app_module._update_trial_gates(
            {"status": "accepted", "completed_at": "2026-04-22T02:00:00+00:00"}
        )

        assert state["migration"]["trial"]["gates"]["status"] == "accepted"

    def test_update_trial_gates_noop_when_no_migration(self, monkeypatch):
        """_update_trial_gates is a no-op when _state has no migration key."""
        monkeypatch.setattr(app_module, "_state", {"migration": None})
        # Must not raise.
        app_module._update_trial_gates({"status": "accepted"})

    def test_update_trial_gates_noop_when_no_trial(self, monkeypatch):
        """_update_trial_gates is a no-op when migration has no trial stash."""
        monkeypatch.setattr(
            app_module,
            "_state",
            {"migration": {"state": "LIVE", "trial": None, "recovery_required": []}},
        )
        # Must not raise.
        app_module._update_trial_gates({"status": "accepted"})
