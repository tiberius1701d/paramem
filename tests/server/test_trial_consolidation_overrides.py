"""Tests for trial consolidation overrides.

Verifies that the trial consolidation loop is configured with:
- mode="train" regardless of the candidate config's consolidation.mode
- output paths pointing to state/trial/adapters/ and state/trial/graph/
- gates set to "no_new_sessions" on empty queue
- gates set to "trial_exception" when the trainer raises

No GPU — all tests use mocked model/tokenizer/consolidation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

    state_dir = tmp_path / "data" / "ha" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    trial_adapter_dir = str((tmp_path / "data" / "ha" / "state" / "trial" / "adapters").resolve())
    trial_graph_dir = str((tmp_path / "data" / "ha" / "state" / "trial" / "graph").resolve())

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

    def test_trial_loop_graph_is_ram_only(self, tmp_path):
        """_build_trial_loop does not set persist_graph or graph_path.

        The trial loop writes no dedicated graph file; _stash_trial_graph stashes
        either the interim-slot graph.json path (simulate) or a reconstructed
        in-memory graph (train) after the fold, not a persisted merger graph.
        """
        state = _make_state(tmp_path)
        trial_adapter_dir = Path(state["migration"]["trial"]["trial_adapter_dir"])
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])

        # Track which attributes are SET (not just accessed) on the mock loop.
        assigned: dict = {}

        class _TrackingMock(MagicMock):
            def __setattr__(self, name, value):
                if not name.startswith("_"):
                    assigned[name] = value
                super().__setattr__(name, value)

        mock_loop = _TrackingMock()

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

        # persist_graph and graph_path must NOT be assigned to the loop.
        assert "persist_graph" not in assigned
        assert "graph_path" not in assigned

    def test_trial_loop_raises_on_malformed_tier_registry(self, tmp_path):
        """A malformed tier registry under trial_adapter_dir propagates loudly.

        _build_trial_loop must NOT swallow a registry load failure and fall
        back to an empty store: a tier's ``indexed_key_registry.json`` that
        exists but is not KeyRegistry-shaped (foreign schema, matching the
        pattern in test_erase_doors.py's
        TestUnreadableTierRegistryAbortsBeforeAnyMutation) must raise
        ValueError naming the offending path, propagated unmodified from
        KeyRegistry.load via MemoryStore.load_registries_from_disk.
        """
        import json

        state = _make_state(tmp_path)
        trial_adapter_dir = Path(state["migration"]["trial"]["trial_adapter_dir"])
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])

        # episodic exists but is not KeyRegistry-shaped (foreign schema:
        # missing 'stale' and 'simhash').
        episodic_dir = trial_adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True, exist_ok=True)
        malformed_registry_path = episodic_dir / "indexed_key_registry.json"
        malformed_registry_path.write_text(json.dumps({"active_keys": ["ghost_key_0"]}))

        with patch("paramem.server.consolidation.create_consolidation_loop") as mock_create:
            from paramem.server.app import _build_trial_loop
            from paramem.server.config import load_server_config

            trial_config = load_server_config(Path(state["config_path"]))

            with pytest.raises(ValueError, match=r"is not a KeyRegistry-shaped") as exc_info:
                _build_trial_loop(
                    state["model"],
                    state["tokenizer"],
                    trial_config,
                    trial_adapter_dir,
                    trial_graph_dir,
                )

        # The raised message names the offending path.
        assert str(malformed_registry_path) in str(exc_info.value)
        # create_consolidation_loop must never be reached — the registry
        # load failure aborts before the loop is constructed.
        mock_create.assert_not_called()


class TestRunTrialConsolidation:
    """Tests for _run_trial_consolidation coroutine.

    All imports inside _run_trial_consolidation are local, so we patch via
    the module where they are defined (paramem.server.consolidation, etc.).

    _run_trial_consolidation calls evaluate_gates internally.
    Tests that only verify outer state transitions (exception handling, gate
    status) patch evaluate_gates at the module boundary to keep them fast and
    isolated from gate internals.
    """

    def test_trial_exception_sets_trial_exception_gate(self, tmp_path, monkeypatch):
        """When _run_trial_consolidation outer try-block raises, gates become trial_exception.

        The outer exception path (e.g. load_server_config failing) sets
        {"status": "trial_exception", ...}.  Gate-level exceptions are handled
        inside evaluate_gates and produce "fail", not "trial_exception".
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

        session_buffer is None → session_buffer_empty=True → all gates
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
    """Regression guard: trial cycle must never call session_buffer.mark_consolidated."""

    def test_trial_cycle_does_not_mark_consolidated(self, tmp_path, monkeypatch):
        """_run_extraction_phase called from trial path must not invoke mark_consolidated.

        Verifies the ``mark_sessions=False`` plumbing: the real
        session_buffer.mark_consolidated must not be called so that pending
        sessions stay in the buffer after the trial cycle.

        session_buffer.pending_count is set to 2 so _run_extraction_phase
        is called (buffer-empty path skips it).  evaluate_gates is patched to
        avoid GPU interactions.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 2  # non-zero → _run_extraction_phase called
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
                            "paramem.server.app._run_extraction_phase",
                            return_value=mock_summary,
                        ) as mock_run:
                            with patch(
                                "paramem.server.gates.evaluate_gates",
                                return_value=skipped_gates,
                            ):
                                await app_module._run_trial_consolidation()
                            # Verify mark_sessions=False was passed.
                            # The trial path must not mark sessions consolidated.
                            call_kwargs = mock_run.call_args.kwargs
                            assert call_kwargs.get("mark_sessions") is False, (
                                "trial path must pass mark_sessions=False so sessions stay pending"
                            )

        asyncio.run(_run())

    def test_trial_pending_count_unchanged_after_trial(self, tmp_path, monkeypatch):
        """After a trial run, session_buffer.get_pending() returns the same list as before.

        The trial cycle must leave the session buffer untouched so
        /migration/rollback (3b.3) finds the original pending queue intact.

        session_buffer.pending_count is set to 2 so _run_extraction_phase
        is called.  evaluate_gates is patched to avoid GPU interactions.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        pending_before = [{"session_id": "s1"}, {"session_id": "s2"}]
        mock_session_buffer = MagicMock()
        mock_session_buffer.pending_count = 2  # non-zero → _run_extraction_phase called
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
                            "paramem.server.app._run_extraction_phase",
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

    def test_trial_loop_uses_configured_mode(self, tmp_path, monkeypatch):
        """Trial consolidation runs in the candidate's CONFIGURED mode — no force-train.

        The former force-train override was removed: a pure consolidation.mode
        change is applied directly by migration_confirm (and rebuilt by the
        active-store migration), so only non-mode changes reach the trial, where
        the live mode is unchanged and the trial must faithfully reflect it.
        Here the candidate is simulate, so the trial must run in simulate mode.
        session_buffer is provided with pending_count > 0 so that
        _run_extraction_phase is actually called (buffer-empty path skips it).
        evaluate_gates is patched to avoid GPU interactions.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import GateResult

        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        # Give the state a non-empty session buffer so _run_extraction_phase is called.
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
                    # Candidate sets mode=simulate — trial must run in simulate (no override).
                    from paramem.server.config import ConsolidationScheduleConfig

                    real_cfg = MagicMock()
                    real_cfg.consolidation = ConsolidationScheduleConfig(mode="simulate")
                    mock_load.return_value = real_cfg

                    with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_gpu:
                        mock_gpu.return_value.__enter__ = MagicMock(return_value=None)
                        mock_gpu.return_value.__exit__ = MagicMock(return_value=False)

                        def capture_run(loop, mark_sessions=True):
                            # _run_extraction_phase reads config from _state;
                            # capture the mode that was injected by _run_trial_consolidation.
                            import paramem.server.app as _app

                            captured_cfg["mode"] = _app._state["config"].consolidation.mode
                            return mock_summary

                        with patch(
                            "paramem.server.app._run_extraction_phase",
                            side_effect=capture_run,
                        ):
                            with patch(
                                "paramem.server.gates.evaluate_gates",
                                return_value=skipped_gates,
                            ):
                                await app_module._run_trial_consolidation()

        asyncio.run(_run())

        assert captured_cfg.get("mode") == "simulate", (
            "trial consolidation must run in the candidate's configured mode (no force-train)"
        )


class TestUpdateTrialGates:
    """Tests for _update_trial_gates helper."""

    def test_update_trial_gates_sets_gate_dict(self, tmp_path, monkeypatch):
        """_update_trial_gates writes into _state["migration"]["trial"]["gates"]."""
        state = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        asyncio.run(
            app_module._update_trial_gates(
                {"status": "accepted", "completed_at": "2026-04-22T02:00:00+00:00"}
            )
        )

        assert state["migration"]["trial"]["gates"]["status"] == "accepted"

    def test_update_trial_gates_noop_when_no_migration(self, monkeypatch):
        """_update_trial_gates is a no-op when _state has no migration key."""

        async def _run():
            monkeypatch.setattr(
                app_module, "_state", {"migration": None, "migration_lock": asyncio.Lock()}
            )
            # Must not raise.
            await app_module._update_trial_gates({"status": "accepted"})

        asyncio.run(_run())

    def test_update_trial_gates_noop_when_no_trial(self, monkeypatch):
        """_update_trial_gates is a no-op when migration has no trial stash."""

        async def _run():
            monkeypatch.setattr(
                app_module,
                "_state",
                {
                    "migration": {"state": "LIVE", "trial": None, "recovery_required": []},
                    "migration_lock": asyncio.Lock(),
                },
            )
            # Must not raise.
            await app_module._update_trial_gates({"status": "accepted"})

        asyncio.run(_run())

    def test_update_trial_gates_blocks_on_cancel(self, tmp_path, monkeypatch):
        """Concurrent cancel-clearing-trial cannot interleave the read-modify-write.

        Holds ``migration_lock`` from a separate task while clearing
        ``_state["migration"]["trial"]``; ``_update_trial_gates`` must wait for
        the lock to release, then observe the cleared state and no-op.
        """

        async def _run():
            state = _make_state(tmp_path)
            monkeypatch.setattr(app_module, "_state", state)
            lock = state["migration_lock"]

            async def _simulate_cancel():
                async with lock:
                    state["migration"]["trial"] = None
                    # Yield so the racing _update_trial_gates is scheduled
                    # and blocks on the lock while we still hold it.
                    await asyncio.sleep(0)

            cancel_task = asyncio.create_task(_simulate_cancel())
            await asyncio.sleep(0)  # let cancel_task acquire the lock first
            await app_module._update_trial_gates({"status": "accepted"})
            await cancel_task

            assert state["migration"]["trial"] is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Gate extension tests — additional gate-override scenarios
# ---------------------------------------------------------------------------


class TestTrialGateExtensions:
    """Gate integration tests for _run_trial_consolidation: payload shape and backward-compat."""

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
        list (backward-compat: trial_exception gates dict is minimal).
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
                            "paramem.server.app._run_extraction_phase",
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
                            "paramem.server.app._run_extraction_phase",
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
# Loud failure when _state["config"] is missing
# ---------------------------------------------------------------------------


class TestRunTrialConsolidationMissingConfig:
    """Verify that _run_trial_consolidation raises loudly when config is absent.

    Production always has _state["config"] set at lifespan startup.  If it is
    somehow missing, a silent Path("state/registry.json") fallback would mask
    the bug, so the handler raises a RuntimeError instead
    (``paramem/server/app.py``, the ``live_config is None`` guard).  The outer
    ``except Exception`` block catches it and surfaces it as
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


class TestRunExtractionPhasePropagatesExtractionFailed:
    """``_run_extraction_phase`` (the trial path's own extract-all, a
    near-duplicate of ``_extract_pending_sessions`` that bypasses
    ``retirable``) has no per-chunk isolation and no abort-reporting
    result field for ``ExtractionFailed`` — unlike ``_extract_pending_sessions``,
    it does not catch it at all, so a local-extraction parse failure must
    propagate straight to the caller. Its only production caller passes
    ``mark_sessions=False`` and wraps the call in its own
    ``except Exception as _exc: exc_captured = _exc`` (the trial dispatch's
    gate machinery, exercised end-to-end elsewhere in this module) — this
    test pins the propagation itself, directly, so a future ``except``
    added inside ``_run_extraction_phase`` cannot silently reinstate a
    swallow."""

    def test_local_extraction_failure_propagates_uncaught(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        from paramem.graph.extractor import ExtractionFailed
        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.server.session_buffer import SessionBuffer

        config = ServerConfig()
        ha = tmp_path / "ha"
        config.paths = PathsConfig(data=ha, sessions=ha / "sessions", debug=ha / "debug")
        (ha / "adapters").mkdir(parents=True, exist_ok=True)

        buffer = SessionBuffer(ha / "sessions", debug=False)
        buffer.set_speaker("s1", "speaker0", "speaker0")
        buffer.append("s1", "user", "Hello, this is a test session.")

        loop = MagicMock()
        loop.shutdown_requested = False
        loop.extract_session = MagicMock(
            side_effect=ExtractionFailed("local_extract", "ValueError: bad json")
        )

        no_lock = MagicMock()
        no_lock.__enter__ = MagicMock(return_value=None)
        no_lock.__exit__ = MagicMock(return_value=False)

        state = {"config": config, "session_buffer": buffer, "speaker_store": None}
        monkeypatch.setattr(app_module, "_state", state)

        with patch("paramem.server.app.vram_scope", return_value=no_lock):
            with pytest.raises(ExtractionFailed) as exc_info:
                app_module._run_extraction_phase(loop, mark_sessions=False)

        assert exc_info.value.phase == "local_extract"
