"""Tests for the process-side VRAM safety net.

Covers the two integration points:
- ``apply_process_cap`` invokes the torch cap with the configured
  fraction on CUDA-available systems and is a no-op otherwise.
- ``vram_scope`` always clears the cache on exit, converts
  ``torch.cuda.OutOfMemoryError`` into :class:`VramExhausted` with the
  phase label attached, and is a transparent no-op when CUDA is
  unavailable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from paramem.server.vram_guard import (
    DEFAULT_PROCESS_FRACTION,
    VramExhausted,
    apply_process_cap,
    check_vram_headroom,
    vram_scope,
)


class TestApplyProcessCap:
    def test_no_op_when_cuda_unavailable(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=False):
            with patch(
                "paramem.server.vram_guard.torch.cuda.set_per_process_memory_fraction"
            ) as cap:
                apply_process_cap()
        cap.assert_not_called()

    def test_default_fraction_applied_on_cuda(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch(
                "paramem.server.vram_guard.torch.cuda.set_per_process_memory_fraction"
            ) as cap:
                apply_process_cap()
        cap.assert_called_once_with(DEFAULT_PROCESS_FRACTION, device=0)

    def test_explicit_fraction_and_device_passed_through(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch(
                "paramem.server.vram_guard.torch.cuda.set_per_process_memory_fraction"
            ) as cap:
                apply_process_cap(fraction=0.7, device=1)
        cap.assert_called_once_with(0.7, device=1)


class TestVramScope:
    def test_no_op_when_cuda_unavailable(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=False):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with vram_scope("s001"):
                    pass
        empty.assert_not_called()

    def test_empty_cache_called_on_clean_exit(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with vram_scope("s001"):
                    pass
        empty.assert_called_once_with()

    def test_oom_converted_to_vram_exhausted(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted) as info:
                    with vram_scope("s042"):
                        raise torch.cuda.OutOfMemoryError("simulated")
        # The phase label is the first arg so callers can distinguish
        # extract sessions ("s042") from the training step ("training").
        assert info.value.args == ("s042",)
        assert isinstance(info.value.__cause__, torch.cuda.OutOfMemoryError)

    def test_empty_cache_called_on_oom(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with pytest.raises(VramExhausted):
                    with vram_scope("s042"):
                        raise torch.cuda.OutOfMemoryError("simulated")
        # One on the OOM path before re-raise, one in the finally — both fine.
        assert empty.call_count >= 1

    def test_non_oom_exception_propagates_unchanged(self):
        # Use a plain RuntimeError whose message does NOT match any
        # CUDA-driver-fault marker — the widened catch must let it pass.
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with pytest.raises(RuntimeError, match="not an OOM"):
                    with vram_scope("s003"):
                        raise RuntimeError("not an OOM")
        empty.assert_called_once_with()

    @pytest.mark.parametrize(
        "marker_message",
        [
            "CUDA driver error: device not ready",
            "CUDA error: device not ready",
            "device not ready",
        ],
    )
    def test_cuda_driver_fault_runtimeerror_converted_to_vram_exhausted(self, marker_message):
        """WSL2 dxgkrnl reports unsatisfiable allocations as bare
        ``RuntimeError("...device not ready...")``, not ``OutOfMemoryError``.

        The vram_scope must treat these as the same VRAM-overalloc class
        so ``last_consolidation_error`` populates and the cycle aborts
        cleanly.
        """
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted) as info:
                    with vram_scope("plaus_filter"):
                        raise RuntimeError(marker_message)
        assert info.value.args == ("plaus_filter",)
        assert isinstance(info.value.__cause__, RuntimeError)

    def test_empty_cache_called_on_cuda_driver_fault(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with pytest.raises(VramExhausted):
                    with vram_scope("plaus_filter"):
                        raise RuntimeError("CUDA driver error: device not ready")
        # Once on the fault path before re-raise, once in the finally — both fine.
        assert empty.call_count >= 1

    def test_empty_cache_failure_is_swallowed(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch(
                "paramem.server.vram_guard.torch.cuda.empty_cache",
                side_effect=RuntimeError("boom"),
            ):
                # Clean path: empty_cache failure must not break the context manager.
                with vram_scope("s007"):
                    pass


class TestCheckVramHeadroom:
    """check_vram_headroom is a drift detector: warn + state, never raise."""

    _HEADROOM = int(1.5 * 2**30)  # 1.5 GiB — matches configs/server.yaml

    def test_no_op_when_cuda_unavailable(self):
        state: dict = {}
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=False):
            check_vram_headroom("s001", self._HEADROOM, state)
        assert "vram_low_headroom_warning" not in state

    def test_silent_when_free_above_threshold(self):
        state: dict = {}
        with (
            patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True),
            patch(
                "paramem.server.vram_guard.torch.cuda.mem_get_info",
                return_value=(3 * 2**30, 8 * 2**30),
            ),
        ):
            check_vram_headroom("s001", self._HEADROOM, state)
        assert "vram_low_headroom_warning" not in state

    def test_populates_state_when_below_threshold(self):
        state: dict = {}
        with (
            patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True),
            patch(
                "paramem.server.vram_guard.torch.cuda.mem_get_info",
                return_value=(1 * 2**30, 8 * 2**30),  # 1 GiB free < 1.5 GiB headroom
            ),
        ):
            check_vram_headroom("s042", self._HEADROOM, state)
        warn = state.get("vram_low_headroom_warning")
        assert warn is not None, "state must be populated when free < headroom"
        assert warn["label"] == "s042"
        assert warn["free_gib"] == pytest.approx(1.0)
        assert warn["headroom_gib"] == pytest.approx(1.5, rel=1e-3)
        assert warn["total_gib"] == pytest.approx(8.0)
        assert "observed_at" in warn

    def test_never_raises_on_low_free(self):
        """The contract: warn, do NOT abort. vram_scope is the actual OOM catch."""
        state: dict = {}
        with (
            patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True),
            patch(
                "paramem.server.vram_guard.torch.cuda.mem_get_info",
                return_value=(64 * 2**20, 8 * 2**30),  # 64 MiB free — well below
            ),
        ):
            # Must NOT raise VramExhausted.
            check_vram_headroom("s003", self._HEADROOM, state)

    def test_mem_get_info_fault_swallowed(self):
        """Driver fault on mem_get_info is dropped — vram_scope handles real OOM."""
        state: dict = {}
        with (
            patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True),
            patch(
                "paramem.server.vram_guard.torch.cuda.mem_get_info",
                side_effect=RuntimeError("driver unhealthy"),
            ),
        ):
            check_vram_headroom("s004", self._HEADROOM, state)
        assert "vram_low_headroom_warning" not in state

    def test_state_optional(self):
        """When no state dict is passed, function still runs (logs only, no state)."""
        with (
            patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True),
            patch(
                "paramem.server.vram_guard.torch.cuda.mem_get_info",
                return_value=(1 * 2**30, 8 * 2**30),
            ),
        ):
            check_vram_headroom("s005", self._HEADROOM, None)


class TestConsolidationIntegration:
    """`_run_extraction_phase` wraps every `extract_session` in `session_guard`.

    A torch OOM during extraction must abort the cycle (raise
    :class:`VramExhausted` outward, do NOT continue to the next session).
    """

    @staticmethod
    def _make_mock_loop():
        from unittest.mock import MagicMock

        loop = MagicMock()
        loop.shutdown_requested = False
        loop.merger = MagicMock()
        loop.merger.graph = MagicMock()
        loop.merger.graph.nodes = []
        loop.promoted_keys = set()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        loop.train_adapters = MagicMock(return_value={})
        loop.cycle_count = 0
        return loop

    @staticmethod
    def _make_config(tmp_path):
        from paramem.server.config import PathsConfig, ServerConfig

        # Override every path that ConsolidationLoop or its callees may
        # write/read. PathsConfig defaults are RELATIVE ("data/ha", etc.) and
        # resolve against cwd, so leaving any of debug/sessions/simulate at
        # default routes test side-effects into the live data dir.
        config = ServerConfig()
        ha = tmp_path / "ha"
        config.paths = PathsConfig(
            data=ha,
            sessions=ha / "sessions",
            debug=ha / "debug",
        )
        (ha / "adapters").mkdir(parents=True, exist_ok=True)
        return config

    @staticmethod
    def _make_session_buffer(tmp_path, conv_id, speaker_id):
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
        buffer.set_speaker(conv_id, speaker_id, speaker_id)
        buffer.append(conv_id, "user", "Hello there")
        buffer.append(conv_id, "assistant", "Hi!")
        return buffer

    @staticmethod
    def _call_run_extraction_phase(loop, config, buffer):
        """Inject config + session_buffer into _state and call _run_extraction_phase.

        run_consolidation was deleted; tests call _run_extraction_phase directly.
        """
        import paramem.server.app as _app

        prior_config = _app._state.get("config")
        prior_buffer = _app._state.get("session_buffer")
        prior_ha = _app._state.get("ha_client")
        prior_speaker = _app._state.get("speaker_store")
        _app._state["config"] = config
        _app._state["session_buffer"] = buffer
        _app._state["ha_client"] = None
        _app._state["speaker_store"] = None
        try:
            return _app._run_extraction_phase(loop)
        finally:
            _app._state["config"] = prior_config
            _app._state["session_buffer"] = prior_buffer
            _app._state["ha_client"] = prior_ha
            _app._state["speaker_store"] = prior_speaker

    def test_oom_during_extract_aborts_cycle(self, tmp_path):
        loop = self._make_mock_loop()
        loop.extract_session = lambda *_a, **_kw: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("simulated")
        )

        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, "conv-vram-1", "Speaker7")

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted):
                    self._call_run_extraction_phase(loop, config, buffer)

    def test_oom_in_first_session_skips_remaining(self, tmp_path):
        from unittest.mock import MagicMock

        loop = self._make_mock_loop()
        # First call raises OOM; if the loop ever called extract a second time,
        # the side-effect list would be exhausted (StopIteration), so this
        # doubles as a "no second call" guard.
        loop.extract_session = MagicMock(side_effect=[torch.cuda.OutOfMemoryError("simulated")])

        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, "conv-vram-2a", "Speaker7")
        buffer.set_speaker("conv-vram-2b", "Speaker7", "Speaker7")
        buffer.append("conv-vram-2b", "user", "Second session")
        buffer.append("conv-vram-2b", "assistant", "Reply")

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted):
                    self._call_run_extraction_phase(loop, config, buffer)

        assert loop.extract_session.call_count == 1

    def test_training_oom_surfaces_training_phase_label(self, tmp_path):
        """M2: vram_scope("training") at callsite 2 must label OOM as phase "training".

        ``_run_extraction_phase`` wraps the ``run_consolidation_cycle`` train call
        in ``vram_scope("training")``.  An OOM during training must raise
        ``VramExhausted("training")`` — not ``VramExhausted("some_session_id")`` —
        so the ``/status`` operator endpoint can distinguish training failures from
        extraction failures.
        """
        from unittest.mock import MagicMock

        loop = self._make_mock_loop()
        # Extraction succeeds; training raises OOM.
        loop.extract_session = MagicMock(
            return_value=(
                [
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Berlin",
                        "relation_type": "factual",
                        "speaker_id": "Speaker7",
                    }
                ],
                [],
            )
        )
        loop.run_consolidation_cycle = MagicMock(
            side_effect=torch.cuda.OutOfMemoryError("training OOM")
        )
        # run_consolidation_cycle is called with mode="train" — the OOM must
        # propagate through vram_scope("training") and surface as VramExhausted.

        config = self._make_config(tmp_path)
        # Add adapters.episodic.enabled to satisfy the guard in _run_extraction_phase.
        config.adapters.episodic.enabled = True
        config.consolidation.mode = "train"
        buffer = self._make_session_buffer(tmp_path, "conv-vram-train-1", "Speaker7")

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted) as exc_info:
                    self._call_run_extraction_phase(loop, config, buffer)

        assert exc_info.value.args == ("training",), (
            f"Expected VramExhausted('training'), got VramExhausted{exc_info.value.args!r}"
        )

    def test_oom_during_training_aborts_cycle(self, tmp_path):
        """OOM in run_consolidation_cycle aborts the cycle as VramExhausted.

        The train callsite in _run_extraction_phase wraps run_consolidation_cycle
        in vram_scope("training") so OutOfMemoryError surfaces as
        VramExhausted("training") — the /status operator endpoint uses the phase
        label to distinguish training failures from extraction failures.
        """
        loop = self._make_mock_loop()
        # Extract returns one QA pair so the cycle reaches the training step.
        loop.extract_session = lambda *_a, **_kw: ([{"key": "k1", "speaker_id": "Speaker7"}], [])
        loop.run_consolidation_cycle = lambda *_a, **_kw: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("training simulated")
        )

        config = self._make_config(tmp_path)
        config.consolidation.mode = "train"
        buffer = self._make_session_buffer(tmp_path, "conv-vram-train", "Speaker7")

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted) as exc_info:
                    self._call_run_extraction_phase(loop, config, buffer)

        assert exc_info.value.args == ("training",), (
            f"Expected VramExhausted('training'), got VramExhausted{exc_info.value.args!r}"
        )


class TestVramConfig:
    """Validator on VramConfig.process_cap_fraction.

    The cap fraction is read at server startup; an invalid value at the
    YAML level is caught at config-load time, not at first allocation.
    """

    def test_default_fraction(self):
        from paramem.server.config import VramConfig

        cfg = VramConfig()
        assert cfg.process_cap_fraction == 0.85

    def test_explicit_valid_fraction(self):
        from paramem.server.config import VramConfig

        cfg = VramConfig(process_cap_fraction=0.9)
        assert cfg.process_cap_fraction == 0.9

    def test_zero_rejected(self):
        from paramem.server.config import VramConfig

        with pytest.raises(ValueError, match="process_cap_fraction"):
            VramConfig(process_cap_fraction=0.0)

    def test_above_one_rejected(self):
        from paramem.server.config import VramConfig

        with pytest.raises(ValueError, match="process_cap_fraction"):
            VramConfig(process_cap_fraction=1.5)

    def test_negative_rejected(self):
        from paramem.server.config import VramConfig

        with pytest.raises(ValueError, match="process_cap_fraction"):
            VramConfig(process_cap_fraction=-0.1)

    def test_yaml_loader_consumes_section(self, tmp_path):
        """`load_server_config` reads `vram:` section into ServerConfig.vram."""
        from paramem.server.config import load_server_config

        cfg_path = tmp_path / "server.yaml"
        cfg_path.write_text("vram:\n  process_cap_fraction: 0.7\n")
        loaded = load_server_config(cfg_path)
        assert loaded.vram.process_cap_fraction == 0.7

    def test_yaml_loader_invalid_fraction_raises(self, tmp_path):
        from paramem.server.config import load_server_config

        cfg_path = tmp_path / "server.yaml"
        cfg_path.write_text("vram:\n  process_cap_fraction: 2.0\n")
        with pytest.raises(ValueError, match="process_cap_fraction"):
            load_server_config(cfg_path)


class TestDoneCallbackErrorSurfacing:
    """`_scheduled_extract_done_callback` populates `_state["last_consolidation_error"]`.

    The callback runs on the asyncio event loop after the consolidation
    executor finishes. When the executor raised :class:`VramExhausted`,
    the callback must stash a structured error in `_state` so /status
    can surface it. Other exceptions are logged but do not populate the
    field.
    """

    @staticmethod
    def _make_future_with_exception(exc):
        from concurrent.futures import Future

        future = Future()
        future.set_exception(exc)
        return future

    def test_vram_exhausted_populates_state(self, tmp_path):
        """On VramExhausted, _scheduled_extract_done_callback records a durable
        incident in the incident store (``incidents.json``) and clears the
        ``consolidating`` flag.  The incident detail matches the historic shape
        ``{"type", "phase", "at"}`` so downstream ``/status`` consumers stay
        HTTP-stable.
        """
        from paramem.server.app import _scheduled_extract_done_callback, _state
        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.server.incidents import read_incidents

        cfg = ServerConfig()
        cfg.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
        )
        prior_config = _state.get("config")
        try:
            _state["config"] = cfg
            future = self._make_future_with_exception(VramExhausted("session-xyz"))
            _scheduled_extract_done_callback(future)
            # Incident must be recorded in the durable store.
            incidents = read_incidents(tmp_path / "state")
            active = [i for i in incidents if i.type == "vram_exhausted" and i.status == "active"]
            assert len(active) == 1, f"Expected one active vram_exhausted incident; got {incidents}"
            inc = active[0]
            # Detail shape matches the historic {type, phase, at} contract.
            assert inc.detail["type"] == "vram_exhausted"
            assert inc.detail["phase"] == "session-xyz"
            assert "at" in inc.detail
            assert _state["consolidating"] is False
        finally:
            _state["config"] = prior_config

    def test_generic_exception_does_not_populate_state(self, tmp_path):
        """Non-VramExhausted exceptions do not record an incident and do not
        populate the incident store.
        """
        from paramem.server.app import _scheduled_extract_done_callback, _state
        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.server.incidents import read_incidents

        cfg = ServerConfig()
        cfg.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
        )
        prior_config = _state.get("config")
        try:
            _state["config"] = cfg
            future = self._make_future_with_exception(RuntimeError("unrelated"))
            _scheduled_extract_done_callback(future)
            # No incident should be written for a generic exception.
            incidents = read_incidents(tmp_path / "state")
            vram_incidents = [i for i in incidents if i.type == "vram_exhausted"]
            assert vram_incidents == [], f"Expected no vram_exhausted incident; got {incidents}"
            assert _state["consolidating"] is False
        finally:
            _state["config"] = prior_config


class TestPerChunkOOMSkip:
    """`_extract_and_start_training` per-chunk VramExhausted skip mechanism.

    A ``VramExhausted`` raised inside the per-chunk wrapping must:
    - be caught locally (not propagate out of the cycle)
    - record the chunk's session_id in the local ``failed_session_ids`` set
    - append a ``{session_id, phase, at}`` entry to ``_state["chunk_failures"]``
    - exclude the failed chunk from the ``mark_consolidated`` call so it
      stays pending for retry on the next cycle
    - allow subsequent chunks in the same cycle to proceed normally

    Mirrors the empirical behaviour the user verified across 3 sequential
    live cycles (post-cycle state was identical 5626/2266 with no
    accumulating leak; the skip path didn't fire because all chunks
    succeeded). Without a unit test, regressions in
    ``_completed_session_ids`` / ``mark_consolidated`` filtering can ship
    silently.
    """

    @staticmethod
    def _make_state(tmp_path):
        """Build a minimal `_state` mock that lets `_extract_and_start_training`
        run end-to-end without touching real GPU work or HF/wyoming.

        Returns (config, buffer, loop) for the caller's assertions.
        """
        from unittest.mock import MagicMock

        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.server.session_buffer import SessionBuffer

        # Two doc-source pending sessions so we can verify subsequent
        # chunks proceed after the first one's OOM.
        config = ServerConfig()
        ha = tmp_path / "ha"
        config.paths = PathsConfig(
            data=ha,
            sessions=ha / "sessions",
            debug=ha / "debug",
        )
        # Disable indexed_key_replay so the no-facts/early-exit path
        # is the simplest and most testable shape (we're only verifying
        # the per-chunk skip + mark_consolidated filter, not training).
        config.consolidation.indexed_key_replay = False

        (ha / "adapters").mkdir(parents=True, exist_ok=True)

        buffer = SessionBuffer(ha / "sessions", state_dir=ha / "state", debug=False)
        # Two pending document sessions, both with speaker_id set.
        for sid in ("doc-aaa", "doc-bbb"):
            buffer.set_speaker(sid, "Speaker1", "Speaker1")
            buffer.append(
                sid,
                "user",
                f"Synthetic chunk for {sid}",
                metadata={"source_type": "document", "doc_title": sid, "chunk_index": 0},
            )

        # Mock loop with extract_session: VramExhausted for doc-aaa,
        # success ([], []) for doc-bbb. Track call count to assert the
        # second chunk really did run.
        loop = MagicMock()
        loop.shutdown_requested = False
        loop.config = MagicMock()
        loop.config.indexed_key_replay_enabled = False

        from paramem.server.vram_guard import VramExhausted as _Exh

        def _extract(transcript, sid, **kwargs):
            if sid == "doc-aaa":
                raise _Exh("doc-aaa")
            return ([], [])

        loop.extract_session = MagicMock(side_effect=_extract)
        loop.model = None  # cloud-only model handle (irrelevant to this path)

        return config, buffer, loop

    def test_first_chunk_oom_lets_second_chunk_run_and_filter_mark_consolidated(self, tmp_path):
        """Single VramExhausted on chunk #1 must NOT poison chunk #2.

        Asserts both `extract_session` calls happen (chunk #2 reaches
        the loop), `_state["chunk_failures"]` records the first chunk,
        and `mark_consolidated` is called with only the survivor.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server import app as app_module

        config, buffer, loop = self._make_state(tmp_path)

        # Patch the SessionBuffer.mark_consolidated so we can inspect args
        # without it having to manage real archive state.
        marked: list[list[str]] = []
        original_mark = buffer.mark_consolidated

        def _capture_mark(ids, *, retention_dir=None):
            marked.append(list(ids))
            original_mark([], retention_dir=retention_dir)

        buffer.mark_consolidated = _capture_mark

        prior_state = {k: app_module._state.get(k) for k in app_module._state}
        try:
            app_module._state["config"] = config
            app_module._state["session_buffer"] = buffer
            app_module._state["consolidation_loop"] = loop
            app_module._state["model"] = None
            app_module._state["tokenizer"] = None
            app_module._state["chunk_failures"] = []
            # No HA, no speaker_store, no background trainer for this test
            app_module._state["ha_client"] = None
            app_module._state["speaker_store"] = None
            # router is populated by lifespan startup in production; the
            # cycle-completion finalize calls router.reload() unconditionally.
            app_module._state["router"] = MagicMock()

            # GPU lock + voice profile helper + vram_scope + check_vram_headroom
            # are no-ops for this test — we only care about the OOM-skip flow.
            no_lock = MagicMock()
            no_lock.__enter__ = MagicMock(return_value=None)
            no_lock.__exit__ = MagicMock(return_value=False)

            with (
                patch("paramem.server.gpu_lock.gpu_lock_sync", return_value=no_lock),
                patch("paramem.server.app._set_voice_pipeline_profile"),
                patch("paramem.server.app.check_vram_headroom"),
                patch("paramem.server.app.vram_scope", return_value=no_lock),
                # `create_consolidation_loop` is lazily imported inside
                # _extract_and_start_training from paramem.server.consolidation;
                # patch it there.
                patch(
                    "paramem.server.consolidation.create_consolidation_loop",
                    return_value=loop,
                ),
            ):
                app_module._extract_and_start_training()

            # Both chunks attempted — chunk #2 reached the loop after #1's OOM.
            assert loop.extract_session.call_count == 2

            # Failure tracked in _state["chunk_failures"].
            failures = app_module._state.get("chunk_failures", [])
            failure_ids = {f["session_id"] for f in failures}
            assert "doc-aaa" in failure_ids, f"doc-aaa should be in chunk_failures, got {failures}"
            for f in failures:
                if f["session_id"] == "doc-aaa":
                    assert f["phase"] == "doc-aaa"
                    assert "at" in f

            # mark_consolidated was called, but ONLY with chunks that
            # didn't fail. doc-aaa must be excluded; doc-bbb included.
            assert marked, "mark_consolidated was never called"
            all_marked = [sid for batch in marked for sid in batch]
            assert "doc-aaa" not in all_marked, (
                f"doc-aaa was incorrectly marked consolidated: {all_marked}"
            )
            assert "doc-bbb" in all_marked, (
                f"doc-bbb should have been marked consolidated, got {all_marked}"
            )
        finally:
            for k, v in prior_state.items():
                app_module._state[k] = v


class TestExtractionFailedAbortsCycle:
    """ExtractionFailed in any chunk aborts the WHOLE cycle.

    Extraction failure (including SOTA-enrichment HTTP 529) must FAIL the entire
    cycle — sessions/graph stay pending, retry scheduled; silently keeping
    pre-enrichment facts would bake degraded triples into the cumulative graph
    permanently.

    Distinct from VramExhausted, which has per-chunk isolation by design
    (resource constraint on a pathologically dense doc shouldn't poison
    unrelated chunks).
    """

    def _make_state(self, tmp_path):
        from unittest.mock import MagicMock

        from paramem.server.config import PathsConfig, ServerConfig
        from paramem.server.session_buffer import SessionBuffer

        config = ServerConfig()
        ha = tmp_path / "ha"
        config.paths = PathsConfig(
            data=ha,
            sessions=ha / "sessions",
            debug=ha / "debug",
        )
        config.consolidation.indexed_key_replay = False
        (ha / "adapters").mkdir(parents=True, exist_ok=True)

        buffer = SessionBuffer(ha / "sessions", state_dir=ha / "state", debug=False)
        for sid in ("doc-aaa", "doc-bbb", "doc-ccc"):
            buffer.set_speaker(sid, "Speaker1", "Speaker1")
            buffer.append(
                sid,
                "user",
                f"Synthetic chunk for {sid}",
                metadata={"source_type": "document", "doc_title": sid, "chunk_index": 0},
            )

        from paramem.graph.extractor import ExtractionFailed

        loop = MagicMock()
        loop.shutdown_requested = False
        loop.config = MagicMock()
        loop.config.indexed_key_replay_enabled = False

        def _extract(transcript, sid, **kwargs):
            if sid == "doc-bbb":
                raise ExtractionFailed("sota_enrich", "cloud 529")
            return ([], [])

        loop.extract_session = MagicMock(side_effect=_extract)
        loop.model = None
        return config, buffer, loop

    def test_extraction_failed_aborts_remainder_and_keeps_all_sessions_pending(self, tmp_path):
        """A SOTA-enrich ExtractionFailed on chunk #2 must:

        - prevent chunk #3 from being extracted (cycle aborts mid-loop)
        - leave NO chunk marked consolidated, including chunk #1 which
          extracted successfully (no partial-CV bake)
        - record the failed chunk in _state["chunk_failures"]
        - record an ``extraction_failed`` incident in the durable incident store
        - clear the consolidating flag
        """
        from unittest.mock import MagicMock, patch

        from paramem.server import app as app_module
        from paramem.server.incidents import read_incidents

        config, buffer, loop = self._make_state(tmp_path)
        # Incident store lives under config.paths.data / "state".
        state_dir = config.paths.data / "state"

        marked: list[list[str]] = []
        original_mark = buffer.mark_consolidated

        def _capture_mark(ids, *, retention_dir=None):
            marked.append(list(ids))
            original_mark([], retention_dir=retention_dir)

        buffer.mark_consolidated = _capture_mark

        prior_state = {k: app_module._state.get(k) for k in app_module._state}
        try:
            app_module._state["config"] = config
            app_module._state["session_buffer"] = buffer
            app_module._state["consolidation_loop"] = loop
            app_module._state["model"] = None
            app_module._state["tokenizer"] = None
            app_module._state["chunk_failures"] = []
            app_module._state["consolidating"] = True
            app_module._state["ha_client"] = None
            app_module._state["speaker_store"] = None
            app_module._state["router"] = MagicMock()

            no_lock = MagicMock()
            no_lock.__enter__ = MagicMock(return_value=None)
            no_lock.__exit__ = MagicMock(return_value=False)

            with (
                patch("paramem.server.gpu_lock.gpu_lock_sync", return_value=no_lock),
                patch("paramem.server.app._set_voice_pipeline_profile"),
                patch("paramem.server.app.check_vram_headroom"),
                patch("paramem.server.app.vram_scope", return_value=no_lock),
                patch(
                    "paramem.server.consolidation.create_consolidation_loop",
                    return_value=loop,
                ),
            ):
                app_module._extract_and_start_training()

            # Chunks before the failure ran (aaa, bbb); chunk after (ccc)
            # MUST NOT have been extracted — cycle aborted at bbb.
            attempted = [call.args[1] for call in loop.extract_session.call_args_list]
            assert attempted == ["doc-aaa", "doc-bbb"], (
                f"Cycle should have aborted after doc-bbb; attempted={attempted}"
            )

            # NO chunk should have been marked consolidated — the
            # successfully-extracted doc-aaa stays pending too.
            all_marked = [sid for batch in marked for sid in batch]
            assert all_marked == [], (
                f"No chunks should be marked consolidated on cycle abort; got {all_marked}"
            )

            # Failure recorded.
            failures = app_module._state.get("chunk_failures", [])
            failure_ids = {f["session_id"] for f in failures}
            assert "doc-bbb" in failure_ids
            for f in failures:
                if f["session_id"] == "doc-bbb":
                    assert f["phase"] == "sota_enrich"
                    assert "529" in f["reason"]

            # Extraction failure recorded as a durable incident.
            incidents = read_incidents(state_dir)
            ef_incidents = [
                i for i in incidents if i.type == "extraction_failed" and i.status == "active"
            ]
            assert len(ef_incidents) >= 1, (
                f"Expected at least one active extraction_failed incident; got {incidents}"
            )
            inc = ef_incidents[0]
            assert inc.detail["type"] == "extraction_failed"
            assert inc.detail["session_id"] == "doc-bbb"
            assert inc.detail["phase"] == "sota_enrich"

            # Consolidating flag cleared.
            assert app_module._state["consolidating"] is False
        finally:
            for k, v in prior_state.items():
                app_module._state[k] = v
