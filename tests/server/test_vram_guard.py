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
        class _Sentinel(RuntimeError):
            pass

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with pytest.raises(_Sentinel):
                    with vram_scope("s003"):
                        raise _Sentinel("not an OOM")
        empty.assert_called_once_with()

    def test_empty_cache_failure_is_swallowed(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch(
                "paramem.server.vram_guard.torch.cuda.empty_cache",
                side_effect=RuntimeError("boom"),
            ):
                # Clean path: empty_cache failure must not break the context manager.
                with vram_scope("s007"):
                    pass


class TestConsolidationIntegration:
    """`run_consolidation` wraps every `extract_session` in `session_guard`.

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
        loop.indexed_key_qa = {}
        loop.key_sessions = {}
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

        config = ServerConfig()
        config.paths = PathsConfig(data=tmp_path / "ha")
        (tmp_path / "ha" / "adapters").mkdir(parents=True, exist_ok=True)
        return config

    @staticmethod
    def _make_session_buffer(tmp_path, conv_id, speaker_id):
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        buffer.set_speaker(conv_id, speaker_id, speaker_id)
        buffer.append(conv_id, "user", "Hello there")
        buffer.append(conv_id, "assistant", "Hi!")
        return buffer

    def test_oom_during_extract_aborts_cycle(self, tmp_path):
        from paramem.server.consolidation import run_consolidation

        loop = self._make_mock_loop()
        loop.extract_session = lambda *_a, **_kw: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("simulated")
        )

        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, "conv-vram-1", "Speaker7")

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted):
                    run_consolidation(
                        model=None,
                        tokenizer=None,
                        config=config,
                        session_buffer=buffer,
                        loop=loop,
                    )

    def test_oom_in_first_session_skips_remaining(self, tmp_path):
        from unittest.mock import MagicMock

        from paramem.server.consolidation import run_consolidation

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
                    run_consolidation(
                        model=None,
                        tokenizer=None,
                        config=config,
                        session_buffer=buffer,
                        loop=loop,
                    )

        assert loop.extract_session.call_count == 1

    def test_oom_during_training_aborts_cycle(self, tmp_path):
        """Training is wrapped in vram_scope("training") just like extract.

        An OOM in train_adapters_no_save must surface as VramExhausted
        with phase label "training" so operators can distinguish a
        training crash from an extract crash.
        """
        from paramem.server.consolidation import run_consolidation

        loop = self._make_mock_loop()
        # Extract returns one QA pair so the cycle reaches the training step.
        loop.extract_session = lambda *_a, **_kw: ([{"key": "k1", "speaker_id": "Speaker7"}], [])
        loop.train_adapters_no_save = lambda *_a, **_kw: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("training simulated")
        )

        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, "conv-vram-train", "Speaker7")

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted) as info:
                    run_consolidation(
                        model=None,
                        tokenizer=None,
                        config=config,
                        session_buffer=buffer,
                        loop=loop,
                    )
        assert info.value.args == ("training",)


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

    def test_vram_exhausted_populates_state(self):
        from paramem.server.app import _scheduled_extract_done_callback, _state

        prior = _state.get("last_consolidation_error")
        try:
            future = self._make_future_with_exception(VramExhausted("session-xyz"))
            _scheduled_extract_done_callback(future)
            err = _state["last_consolidation_error"]
            assert err is not None
            assert err["type"] == "vram_exhausted"
            assert err["phase"] == "session-xyz"
            assert "at" in err
            assert _state["consolidating"] is False
        finally:
            _state["last_consolidation_error"] = prior

    def test_generic_exception_does_not_populate_state(self):
        from paramem.server.app import _scheduled_extract_done_callback, _state

        prior = _state.get("last_consolidation_error")
        _state["last_consolidation_error"] = None
        try:
            future = self._make_future_with_exception(RuntimeError("unrelated"))
            _scheduled_extract_done_callback(future)
            assert _state["last_consolidation_error"] is None
            assert _state["consolidating"] is False
        finally:
            _state["last_consolidation_error"] = prior
