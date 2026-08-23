"""Shared extraction-stage lifecycle: voice eviction, retirement authority,
and incident/attention authority around ``_extract_pending_sessions`` /
``_extract_and_start_training`` — the surface both a fold's interim tick and
``/calibrate/extract_pending`` reach through the same function.

Fixture pattern mirrors the (now-superseded) ``tests/server/
test_voice_profile_consolidation_paths.py``: a MagicMock ``ConsolidationLoop``
whose ``extract_session`` returns empty relation lists (the no-facts fast
path), a mocked ``SessionBuffer``, and the real ``gpu_lock_sync`` +
``_set_voice_pipeline_profile`` spy pattern — adapted to the current design
(``get_or_create_consolidation_loop``, ``take_pending_relations`` /
``PendingRelations``, ``arbitrate_enrichment_incidents``).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from paramem.training.consolidation import PendingRelations


def _make_pending(source_type: str = "transcript", n: int = 1) -> list[dict]:
    return [
        {
            "session_id": f"sid-{i}",
            "transcript": "Hello",
            "speaker_id": "spk-1",
            "source_type": source_type,
            "doc_title": None,
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:05:00+00:00",
        }
        for i in range(n)
    ]


def _make_loop_no_qa() -> MagicMock:
    """Loop that returns empty relations, so the no-facts early exit fires."""
    loop = MagicMock()
    loop.shutdown_requested = False
    loop.config.consolidation.mode = "train"
    loop.extract_session.return_value = ([], [])
    loop.take_pending_relations.return_value = PendingRelations(episodic=[], procedural=[])
    return loop


def _make_config(mode: str = "train", tmp_path=None) -> MagicMock:
    cfg = MagicMock()
    cfg.adapters.episodic.enabled = True
    cfg.consolidation.mode = mode
    cfg.consolidation.extraction_enrichment_provider = False
    cfg.consolidation.extraction_enrichment_provider_model = ""
    cfg.consolidation.extraction_enrichment_provider_endpoint = ""
    cfg.consolidation.extraction_plausibility_judge = False
    cfg.consolidation.extraction_plausibility_stage = "post"
    cfg.debug = False
    if tmp_path is not None:
        cfg.paths.data = tmp_path
    return cfg


def _make_state_patch(pending: list[dict], config) -> tuple:
    mock_buffer = MagicMock()
    mock_buffer.get_pending.return_value = pending
    mock_buffer.pending_count = len(pending)
    mock_buffer.mark_consolidated.return_value = None
    mock_buffer.pending_facts.return_value = [
        {
            "session_id": s["session_id"],
            "speaker_id": s["speaker_id"],
            "has_voice_embedding": False,
            "age_seconds": 0,
        }
        for s in pending
    ]
    mock_buffer.retirable.side_effect = lambda ids: set(ids)

    from paramem.memory.store import MemoryStore as _MS

    state_patch = {
        "config": config,
        "session_buffer": mock_buffer,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "consolidation_loop": None,
        "memory_store": _MS(),
        "ha_client": None,
        "speaker_store": None,
        "consolidating": True,
        "mode": "local",
        "voice_profile": "gpu",
        "chunk_failures": [],
        "router": MagicMock(),
        "event_loop": None,
    }
    return state_patch, mock_buffer


@contextmanager
def _patch_extract_training(pending, config, loop):
    """Patch ``_state`` for ``_extract_and_start_training`` calls."""
    import paramem.server.app as app_module

    state_patch, mock_buffer = _make_state_patch(pending, config)

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._set_voice_pipeline_profile") as mock_profile,
        patch("paramem.server.app.get_or_create_consolidation_loop", return_value=loop),
        patch("paramem.server.app.check_vram_headroom"),
        patch("paramem.server.app.vram_scope") as mock_vram_scope,
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_vram_scope.return_value.__enter__ = MagicMock(return_value=None)
        mock_vram_scope.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_profile, mock_buffer


# ---------------------------------------------------------------------------
# The no-facts arm of _extract_and_start_training retires exactly
# the sessions it successfully extracted, through _retire_extracted_sessions
# — the same primitive the no-staging terminal (_finalize_interim) calls for
# an identical shape, so both retire the identical set for identical input
# by construction (structural pin below, TestSharedRetirementPrimitive).
# ---------------------------------------------------------------------------


class TestNoFactsArmRetiresThroughTheSharedPrimitive:
    def test_no_facts_batch_is_retired_via_mark_consolidated(self, tmp_path) -> None:
        import paramem.server.app as app_module

        pending = _make_pending(n=2)
        config = _make_config(tmp_path=tmp_path)
        loop = _make_loop_no_qa()

        with _patch_extract_training(pending, config, loop) as (_mock_profile, mock_buffer):
            app_module._extract_and_start_training()

        # mark_consolidated is called once with the retired id set.
        assert len(mock_buffer.mark_consolidated.call_args_list) == 1
        (call_args,) = mock_buffer.mark_consolidated.call_args_list
        retired = set(call_args.args[0])
        assert retired == {"sid-0", "sid-1"}


class TestSharedRetirementPrimitive:
    """Structural pin: the no-facts arm, the no-staging terminal, and the
    full cycle's own consume-pending noop all call the SAME function on the
    SAME argument name -- so "retire the identical set for identical input"
    holds by construction, not by independently-written retirement bodies
    that happen to agree today."""

    def test_retire_extracted_sessions_has_exactly_three_call_sites(self) -> None:
        import ast
        import inspect

        import paramem.server.app as app_module

        source = inspect.getsource(app_module)
        tree = ast.parse(source)
        hits: list[str] = []

        class _Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self._stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self._stack.append(node.name)
                self.generic_visit(node)
                self._stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                if isinstance(node.func, ast.Name) and node.func.id == "_retire_extracted_sessions":
                    hits.append(self._stack[-1] if self._stack else "<module>")
                self.generic_visit(node)

        _Visitor().visit(tree)
        assert set(hits) == {
            "_extract_and_start_training",
            "_finalize_interim",
            "_run_full_cycle",
        }, hits


# ---------------------------------------------------------------------------
# _end_voice_eviction on a run that never evicted is a no-op.
# ---------------------------------------------------------------------------


class TestEndVoiceEvictionNoOpWhenAlreadyAtTarget:
    def test_no_op_when_current_profile_already_matches_target(self, caplog) -> None:
        """A genuine no-op (the idempotent early-return inside
        ``_set_voice_pipeline_profile``), not merely an exception that
        ``_end_voice_eviction``'s own ``try/except`` swallowed and logged --
        that failure mode would leave ``voice_profile`` unchanged too, so
        the state-unchanged assertion alone cannot distinguish the two.
        ``_end_voice_eviction`` is unconditional at every call site (its own
        contract), so it must still have called
        ``_set_voice_pipeline_profile`` exactly once, with the target
        profile -- and that call must not have raised.
        """
        import logging

        import paramem.server.app as app_module

        state = {"mode": "local", "voice_profile": "gpu"}  # already at _target_profile()
        with patch.object(app_module, "_state", state):
            with patch.object(
                app_module,
                "_set_voice_pipeline_profile",
                wraps=app_module._set_voice_pipeline_profile,
            ) as spy:
                with caplog.at_level(logging.ERROR, logger="paramem.server.app"):
                    app_module._end_voice_eviction(lock_held=False)

        spy.assert_called_once_with("gpu", lock_held=False)
        assert not any("Voice restore raised" in r.message for r in caplog.records), (
            "a genuine no-op must not raise (and be silently swallowed) inside "
            "_set_voice_pipeline_profile"
        )
        assert state["voice_profile"] == "gpu"


# ---------------------------------------------------------------------------
# The terminal clears the flag when its own bookkeeping body
# raises, and the exception still propagates.
# ---------------------------------------------------------------------------


class TestTerminalClearsFlagOnRaisingBody:
    def test_flag_clears_even_when_body_raises_and_the_exception_propagates(self) -> None:
        import paramem.server.app as app_module

        state = {"consolidating": True, "event_loop": None}

        def _raising_body() -> None:
            raise RuntimeError("bookkeeping boom")

        with patch.object(app_module, "_state", state):
            try:
                app_module._consolidation_terminal(_raising_body)
            except RuntimeError as exc:
                assert "bookkeeping boom" in str(exc)
            else:
                raise AssertionError("expected RuntimeError to propagate")

        assert state["consolidating"] is False


# ---------------------------------------------------------------------------
# The executor future's done callback is a no-op on a clean
# return (no exception on the future).
# ---------------------------------------------------------------------------


class TestConsolidationRunDoneNoOpOnCleanReturn:
    def test_no_incident_and_no_terminal_dispatch_when_future_has_no_exception(self) -> None:
        import paramem.server.app as app_module
        from paramem.server.consolidation_action import ConsolidationAction

        state = {"consolidating": True, "event_loop": None}
        future = MagicMock()
        future.exception.return_value = None

        with (
            patch.object(app_module, "_state", state),
            patch("paramem.server.app.record_incident") as mock_record,
            patch.object(app_module, "_consolidation_terminal") as mock_terminal,
            patch.object(app_module, "_end_voice_eviction") as mock_evict,
        ):
            app_module._consolidation_run_done(ConsolidationAction.INTERIM, None, future)

        mock_record.assert_not_called()
        mock_terminal.assert_not_called()
        mock_evict.assert_not_called()
        # The flag is untouched by this function on a clean return -- the
        # run's OWN terminal (already dispatched before the future
        # resolved) owns the clear.
        assert state["consolidating"] is True


# ---------------------------------------------------------------------------
# Every submission through _dispatch_to_executor evicts the GPU voice pair
# before the run's own entry point runs -- one envelope, so a staging action
# and a calibrate action see the identical VRAM regime.
# ---------------------------------------------------------------------------


class TestDispatchToExecutorEvictsVoiceBeforeEntryPoint:
    def _dispatch_and_run_submission(self, *, action, spec=None):
        """Call ``_dispatch_to_executor`` against a fake event loop that
        records the submitted callable instead of running it, then invoke
        that callable directly (as the executor thread would) and return the
        call order recorded by the ``_set_voice_pipeline_profile`` /
        entry-point spies.
        """
        import paramem.server.app as app_module

        calls: list[tuple] = []
        loop = MagicMock()
        submitted: list = []

        def _submit(executor, fn):
            submitted.append(fn)
            future = MagicMock()
            future.add_done_callback.return_value = None
            return future

        loop.run_in_executor.side_effect = _submit
        state = {"consolidating": False, "event_loop": loop}

        def _fake_evict(profile, *, lock_held=False) -> None:
            calls.append(("evict", profile, lock_held))

        def _entry() -> None:
            calls.append(("entry",))

        with (
            patch.object(app_module, "_state", state),
            patch.object(app_module, "_set_voice_pipeline_profile", _fake_evict),
        ):
            status = app_module._dispatch_to_executor(_entry, "started", action=action, spec=spec)
            assert len(submitted) == 1, "exactly one callable submitted to the executor"
            submitted[0]()  # run it, as the executor thread would -- still inside the
            # patched scope, since the real _set_voice_pipeline_profile must never run
            # against a fake _state.
        return status, calls

    def test_staging_action_runs_voice_eviction_before_the_entry_point(self) -> None:
        from paramem.server.consolidation_action import ConsolidationAction

        status, calls = self._dispatch_and_run_submission(action=ConsolidationAction.INTERIM)

        assert status == "started"
        assert calls == [("evict", "cpu", False), ("entry",)], (
            "voice must be evicted before the staging entry point runs"
        )

    def test_calibrate_action_runs_voice_eviction_before_the_entry_point(self) -> None:
        from paramem.server.consolidation_action import ConsolidationAction

        status, calls = self._dispatch_and_run_submission(
            action=ConsolidationAction.CALIBRATE, spec=MagicMock()
        )

        assert status == "started"
        assert calls == [("evict", "cpu", False), ("entry",)], (
            "voice must be evicted before the calibrate entry point runs"
        )

    def test_eviction_failure_propagates_and_never_reaches_the_entry_point(self) -> None:
        """``_run_evicted`` adds no exception handling of its own (its own
        documented contract): an eviction failure must propagate to the
        caller UNCHANGED, and the entry point it wraps must never run --
        the executor future then carries the eviction's own exception,
        exactly as if this wrapper did not exist."""
        import paramem.server.app as app_module

        entry_calls: list[str] = []

        def _entry() -> None:
            entry_calls.append("entry")

        def _raising_evict(profile, *, lock_held=False) -> None:
            raise RuntimeError("voice eviction boom")

        with patch.object(app_module, "_set_voice_pipeline_profile", _raising_evict):
            with pytest.raises(RuntimeError, match="voice eviction boom"):
                app_module._run_evicted(_entry)

        assert entry_calls == [], "the entry point must never run when eviction raises"


# ---------------------------------------------------------------------------
# 44 — enrichment_degraded / vram-headroom-attention authority.
# Structural: the extract_pending route's own dispatch closure never
# arbitrates enrichment incidents or copies a vram-headroom warning into
# _state, unlike the two staging callers, which both do.
# extraction_failed stays unconditional -- a direct behavioural check on the
# shared _extract_pending_sessions function.
# ---------------------------------------------------------------------------


class TestEnrichmentAndAttentionAuthorityIsStagingOnly:
    def _find_function(self, name: str):
        import ast
        import inspect

        import paramem.server.app as app_module

        tree = ast.parse(inspect.getsource(app_module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        raise AssertionError(f"{name} not found")

    @staticmethod
    def _calls_arbitrate_enrichment_incidents(func) -> bool:
        import ast

        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "arbitrate_enrichment_incidents"
            for node in ast.walk(func)
        )

    @staticmethod
    def _references_vram_low_headroom_warning(func) -> bool:
        import ast

        return any(
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "vram_low_headroom_warning"
            for node in ast.walk(func)
        )

    def test_extract_pending_route_never_arbitrates_enrichment_incidents(self) -> None:
        route = self._find_function("calibrate_extract_pending_route")
        assert self._calls_arbitrate_enrichment_incidents(route) is False

    def test_extract_pending_route_never_writes_the_vram_headroom_attention_row(self) -> None:
        route = self._find_function("calibrate_extract_pending_route")
        assert self._references_vram_low_headroom_warning(route) is False

    def test_extract_and_start_training_arbitrates_and_adopts_the_headroom_row(self) -> None:
        staging_caller = self._find_function("_extract_and_start_training")
        assert self._calls_arbitrate_enrichment_incidents(staging_caller) is True
        assert self._references_vram_low_headroom_warning(staging_caller) is True

    def test_extraction_failed_incident_is_unconditional_in_the_shared_stage(
        self, tmp_path
    ) -> None:
        """extraction_failed is recorded from inside the ONE shared
        ``_extract_pending_sessions`` function -- reached identically by a
        fold's interim tick and by ``/calibrate/extract_pending`` -- so
        "every run" writes it by construction (there is no caller-identity
        branch inside the function)."""
        import paramem.server.app as app_module
        from paramem.graph.extractor import ExtractionFailed

        pending = _make_pending(n=1)
        config = _make_config(tmp_path=tmp_path)
        loop = MagicMock()
        loop.shutdown_requested = False

        def _raise(*_args, **_kwargs):
            raise ExtractionFailed("local_extract", "bad json")

        loop.extract_session.side_effect = _raise
        loop.take_pending_relations.return_value = PendingRelations(episodic=[], procedural=[])

        state_patch, _mock_buffer = _make_state_patch(pending, config)
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope") as mock_vram_scope,
            patch("paramem.server.app.record_incident") as mock_record_incident,
        ):
            mock_vram_scope.return_value.__enter__ = MagicMock(return_value=None)
            mock_vram_scope.return_value.__exit__ = MagicMock(return_value=False)
            result = app_module._extract_pending_sessions(loop, lock_held=True)

        assert result.aborted is not None
        recorded_types = {
            kwargs.get("type") for _args, kwargs in (c for c in mock_record_incident.call_args_list)
        }
        assert "extraction_failed" in recorded_types


# ---------------------------------------------------------------------------
# No calibrate run retires a session: a second, identical call to
# the shared ``_extract_pending_sessions`` (exactly what ``/calibrate/
# extract_pending``'s own dispatch closure does — see
# ``calibrate_extract_pending_route``) extracts the SAME sessions again,
# because nothing in the function itself ever retires.
# ---------------------------------------------------------------------------


class TestCalibrateNeverRetiresASession:
    def test_two_consecutive_extract_pending_style_calls_see_the_same_sessions(
        self, tmp_path
    ) -> None:
        import paramem.server.app as app_module
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        buffer.append("conv-1", "user", "Hello", speaker_id="speaker1")
        buffer.append("conv-1", "assistant", "Hi")
        expected_session_id = buffer.get_pending()[0]["session_id"]

        config = _make_config(tmp_path=tmp_path)
        loop = MagicMock()
        loop.shutdown_requested = False
        loop.extract_session.return_value = ([], [])
        loop.take_pending_relations.return_value = PendingRelations(episodic=[], procedural=[])

        state_patch = {
            "config": config,
            "session_buffer": buffer,
            "speaker_store": None,
            "chunk_failures": [],
            "event_loop": None,
        }
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope") as mock_vram_scope,
        ):
            mock_vram_scope.return_value.__enter__ = MagicMock(return_value=None)
            mock_vram_scope.return_value.__exit__ = MagicMock(return_value=False)

            first = app_module._extract_pending_sessions(loop, lock_held=True)
            second = app_module._extract_pending_sessions(loop, lock_held=True)

        assert first.session_ids == [expected_session_id]
        assert second.session_ids == [expected_session_id], (
            "a second identical extraction sees the same pending session -- "
            "nothing retired it in between"
        )
        # The session is still on disk, pending -- a calibrate run wrote no
        # retention side effect at all.
        assert [s["session_id"] for s in buffer.get_pending()] == [expected_session_id]
