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

    Slice 4 note: _run_trial_consolidation now calls evaluate_gates internally.
    Tests that only verify outer state transitions (exception handling, gate
    status) patch evaluate_gates at the module boundary to keep them fast and
    isolated from gate internals.
    """

    def test_trial_exception_sets_trial_exception_gate(self, tmp_path, monkeypatch):
        """When _run_trial_consolidation outer try-block raises, gates become trial_exception.

        Slice 4: the outer exception path (e.g. load_server_config failing) is
        preserved — sets {"status": "trial_exception", ...}.  Gate-level exceptions
        are handled inside evaluate_gates and produce "fail", not "trial_exception".
        """
        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        async def _run():
            # Raise at load_server_config — before evaluate_gates is called.
            with patch(
                "paramem.server.config.load_server_config",
                side_effect=RuntimeError("config load OOM"),
            ):
                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "trial_exception"
        assert "exception" in gates

    def test_trial_completion_sets_no_new_sessions_gate(self, tmp_path, monkeypatch):
        """Successful trial with empty queue → gates.status == no_new_sessions.

        Slice 4: session_buffer is None → session_buffer_empty=True → all gates
        return skipped → rollup is no_new_sessions.  evaluate_gates is patched to
        return controlled GateResult objects so this test stays GPU-free.
        """
        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        skipped_gates = [
            GateResult(gate=i, name=n, status="skipped", reason="no_new_sessions", metrics=None)
            for i, n in enumerate(
                ["extraction", "training", "adapter_reload", "live_registry_recall"], start=1
            )
        ]

        async def _run():
            with patch("paramem.server.config.load_server_config") as mock_load:
                cfg = MagicMock()
                cfg.consolidation.mode = "simulate"
                mock_load.return_value = cfg

                with patch("paramem.server.gates.evaluate_gates", return_value=skipped_gates):
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

        Slice 4: session_buffer.pending_count is set to 2 so run_consolidation
        is called (buffer-empty path skips it).  evaluate_gates is patched to
        avoid GPU interactions.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 2  # non-zero → run_consolidation called
        mock_session_buffer.get_pending.return_value = []  # empty queue → no_pending
        state["session_buffer"] = mock_session_buffer

        mock_summary = {"status": "no_pending", "sessions": 0}
        skipped_gates = [
            GateResult(gate=i, name=n, status="skipped", reason="no_new_sessions", metrics=None)
            for i, n in enumerate(
                ["extraction", "training", "adapter_reload", "live_registry_recall"], start=1
            )
        ]

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
                            with patch(
                                "paramem.server.gates.evaluate_gates",
                                return_value=skipped_gates,
                            ):
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

        Slice 4: session_buffer.pending_count is set to 2 so run_consolidation
        is called.  evaluate_gates is patched to avoid GPU interactions.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        pending_before = [{"session_id": "s1"}, {"session_id": "s2"}]
        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 2  # non-zero → run_consolidation called
        mock_session_buffer.get_pending.return_value = pending_before
        state["session_buffer"] = mock_session_buffer

        mock_summary = {"status": "no_pending", "sessions": 0}
        skipped_gates = [
            GateResult(gate=i, name=n, status="skipped", reason="no_new_sessions", metrics=None)
            for i, n in enumerate(
                ["extraction", "training", "adapter_reload", "live_registry_recall"], start=1
            )
        ]

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
                            with patch(
                                "paramem.server.gates.evaluate_gates",
                                return_value=skipped_gates,
                            ):
                                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        # mark_consolidated must NOT have been called on the buffer.
        mock_session_buffer.mark_consolidated.assert_not_called()

    def test_trial_loop_forces_train_mode(self, tmp_path, monkeypatch):
        """Trial consolidation forces consolidation.mode='train' even when candidate says simulate.

        Spec Resolved Decision 27, L239: trial mode is always train.
        Slice 4: session_buffer is provided with pending_count > 0 so that
        run_consolidation is actually called (buffer-empty path skips it).
        evaluate_gates is patched to avoid GPU interactions.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        # Give the state a non-empty session buffer so run_consolidation is called.
        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 2
        state["session_buffer"] = mock_session_buffer

        captured_cfg = {}

        mock_summary = {"status": "no_pending", "sessions": 0}

        skipped_gates = [
            GateResult(gate=i, name=n, status="skipped", reason="no_new_sessions", metrics=None)
            for i, n in enumerate(
                ["extraction", "training", "adapter_reload", "live_registry_recall"], start=1
            )
        ]

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
                            with patch(
                                "paramem.server.gates.evaluate_gates",
                                return_value=skipped_gates,
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


# ---------------------------------------------------------------------------
# Slice 4 extension tests (spec §Tests — 4 extensions)
# ---------------------------------------------------------------------------


class TestSlice4GateExtensions:
    """Slice 4 extension tests for _run_trial_consolidation gate integration."""

    def test_trial_gates_no_new_sessions_has_details_list(self, tmp_path, monkeypatch):
        """status==no_new_sessions AND gates['details'] has exactly 4 entries.

        Verifies that even when all gates are skipped (NO_NEW_SESSIONS), the
        gates payload includes a 'details' key containing 4 GateResult dicts.
        """
        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        skipped_gates = [
            GateResult(gate=i, name=n, status="skipped", reason="no_new_sessions", metrics=None)
            for i, n in enumerate(
                ["extraction", "training", "adapter_reload", "live_registry_recall"], start=1
            )
        ]

        async def _run():
            with patch("paramem.server.config.load_server_config") as mock_load:
                cfg = MagicMock()
                cfg.consolidation.mode = "simulate"
                mock_load.return_value = cfg

                with patch("paramem.server.gates.evaluate_gates", return_value=skipped_gates):
                    await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "no_new_sessions"
        assert "details" in gates, "gates payload must include 'details' list"
        assert len(gates["details"]) == 4, "details must contain exactly 4 GateResult dicts"
        assert all(d["gate"] == i + 1 for i, d in enumerate(gates["details"]))

    def test_trial_gates_trial_exception_has_no_details_list(self, tmp_path, monkeypatch):
        """When outer catch fires (trial_exception), gates does NOT contain 'details' key.

        The outer exception path (e.g. config load failure) sets a minimal gate
        dict with only 'status', 'exception', and 'completed_at' — no 'details'
        list, preserving backward-compat with Slice 3b.3 consumers.
        """
        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        async def _run():
            with patch(
                "paramem.server.config.load_server_config",
                side_effect=RuntimeError("catastrophic failure"),
            ):
                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "trial_exception"
        assert "details" not in gates, (
            "trial_exception path must not include 'details' (no GateResult list generated)"
        )

    def test_trial_gates_pass_rollup_with_full_queue(self, tmp_path, monkeypatch):
        """Fabricated PASS scenario: full queue + adapter files + high-confidence probe.

        Provides a mock session buffer (pending_count > 0) and fabricated
        GateResult objects that represent a full PASS rollup.
        """
        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 3
        state["session_buffer"] = mock_session_buffer

        pass_gates = [
            GateResult(gate=1, name="extraction", status="pass", reason=None, metrics=None),
            GateResult(gate=2, name="training", status="pass", reason=None, metrics=None),
            GateResult(gate=3, name="adapter_reload", status="pass", reason=None, metrics=None),
            GateResult(
                gate=4,
                name="live_registry_recall",
                status="pass",
                reason=None,
                metrics={
                    "recalled": 20,
                    "sampled": 20,
                    "sampled_keys": [f"graph{i}" for i in range(1, 21)],
                    "seed": "abcd1234abcd1234",
                    "retried": False,
                    "warnings": [],
                    "first_sample_recalled": None,
                    "first_sample_seed": None,
                    "first_sample_keys": None,
                },
            ),
        ]

        mock_summary = {"status": "complete", "sessions": 3}

        async def _run():
            mock_loop = MagicMock()
            with patch(
                "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    cfg = MagicMock()
                    cfg.consolidation.mode = "train"
                    mock_load.return_value = cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        with patch(
                            "paramem.server.consolidation.run_consolidation",
                            return_value=mock_summary,
                        ):
                            with patch(
                                "paramem.server.gates.evaluate_gates", return_value=pass_gates
                            ):
                                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "pass"
        assert "details" in gates
        assert len(gates["details"]) == 4
        assert all(d["status"] == "pass" for d in gates["details"])

    def test_trial_gates_fail_on_gate4_miss(self, tmp_path, monkeypatch):
        """FAIL rollup when gate 4 misses on both samples.

        Provides a mock session buffer and fabricated GateResult objects where
        gates 1/2/3 pass but gate 4 fails (5/20 missed twice).  Rollup must be
        'fail'.
        """
        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 3
        state["session_buffer"] = mock_session_buffer

        fail_gates = [
            GateResult(gate=1, name="extraction", status="pass", reason=None, metrics=None),
            GateResult(gate=2, name="training", status="pass", reason=None, metrics=None),
            GateResult(gate=3, name="adapter_reload", status="pass", reason=None, metrics=None),
            GateResult(
                gate=4,
                name="live_registry_recall",
                status="fail",
                reason="recall below threshold on both samples: first=15/20, retry=15/20",
                metrics={
                    "recalled": 15,
                    "sampled": 20,
                    "sampled_keys": [f"graph{i}" for i in range(1, 21)],
                    "seed": "deadbeefdeadbeef",
                    "retried": True,
                    "warnings": [],
                    "first_sample_recalled": 15,
                    "first_sample_seed": "cafebabecafebabe",
                    "first_sample_keys": [f"graph{i}" for i in range(1, 21)],
                },
            ),
        ]

        mock_summary = {"status": "complete", "sessions": 3}

        async def _run():
            mock_loop = MagicMock()
            with patch(
                "paramem.server.consolidation.create_consolidation_loop", return_value=mock_loop
            ):
                with patch("paramem.server.config.load_server_config") as mock_load:
                    cfg = MagicMock()
                    cfg.consolidation.mode = "train"
                    mock_load.return_value = cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        with patch(
                            "paramem.server.consolidation.run_consolidation",
                            return_value=mock_summary,
                        ):
                            with patch(
                                "paramem.server.gates.evaluate_gates", return_value=fail_gates
                            ):
                                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "fail"
        assert gates["details"][3]["status"] == "fail"


# ---------------------------------------------------------------------------
# Fix 4 — loud failure when _state["config"] is missing
# ---------------------------------------------------------------------------


class TestRunTrialConsolidationMissingConfig:
    """Verify that _run_trial_consolidation raises loudly when config is absent.

    Production always has _state["config"] set at lifespan startup.  If it is
    somehow missing the silent Path("state/registry.json") fallback masks the
    bug.  Fix 4 replaces that fallback with a RuntimeError that is caught by
    the outer ``except Exception`` block and surfaces as
    ``gates["status"] == "trial_exception"``.
    """

    def test_missing_config_produces_trial_exception(self, tmp_path, monkeypatch):
        """When _state['config'] is None, gates become trial_exception with message.

        The outer ``except Exception`` block in ``_run_trial_consolidation``
        must catch the RuntimeError from the missing-config guard and write
        ``{"status": "trial_exception", "exception": <message>}`` into the
        gates stash.
        """
        state = _make_state(tmp_path)
        # Remove the config to trigger the loud failure path.
        state["config"] = None
        monkeypatch.setattr(app_module, "_state", state)

        async def _run():
            with patch("paramem.server.config.load_server_config") as mock_load:
                cfg = MagicMock()
                cfg.consolidation.mode = "simulate"
                mock_load.return_value = cfg

                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        gates = state["migration"]["trial"]["gates"]
        assert gates["status"] == "trial_exception"
        assert "exception" in gates
        assert "_state['config'] is missing" in gates["exception"]
