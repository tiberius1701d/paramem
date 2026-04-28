"""Tests for the process-side VRAM safety net.

Covers the two integration points:
- ``apply_process_cap`` invokes the torch cap with the configured
  fraction on CUDA-available systems and is a no-op otherwise.
- ``session_guard`` always clears the cache on exit, converts
  ``torch.cuda.OutOfMemoryError`` into :class:`VramExhausted` with the
  offending session id attached, and is a transparent no-op when CUDA
  is unavailable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from paramem.server.vram_guard import (
    DEFAULT_PROCESS_FRACTION,
    VramExhausted,
    apply_process_cap,
    session_guard,
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


class TestSessionGuard:
    def test_no_op_when_cuda_unavailable(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=False):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with session_guard("s001"):
                    pass
        empty.assert_not_called()

    def test_empty_cache_called_on_clean_exit(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with session_guard("s001"):
                    pass
        empty.assert_called_once_with()

    def test_oom_converted_to_vram_exhausted(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache"):
                with pytest.raises(VramExhausted) as info:
                    with session_guard("s042"):
                        raise torch.cuda.OutOfMemoryError("simulated")
        assert "s042" in str(info.value)
        assert isinstance(info.value.__cause__, torch.cuda.OutOfMemoryError)

    def test_empty_cache_called_on_oom(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with pytest.raises(VramExhausted):
                    with session_guard("s042"):
                        raise torch.cuda.OutOfMemoryError("simulated")
        # One on the OOM path before re-raise, one in the finally — both fine.
        assert empty.call_count >= 1

    def test_non_oom_exception_propagates_unchanged(self):
        class _Sentinel(RuntimeError):
            pass

        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch("paramem.server.vram_guard.torch.cuda.empty_cache") as empty:
                with pytest.raises(_Sentinel):
                    with session_guard("s003"):
                        raise _Sentinel("not an OOM")
        empty.assert_called_once_with()

    def test_empty_cache_failure_is_swallowed(self):
        with patch("paramem.server.vram_guard.torch.cuda.is_available", return_value=True):
            with patch(
                "paramem.server.vram_guard.torch.cuda.empty_cache",
                side_effect=RuntimeError("boom"),
            ):
                # Clean path: empty_cache failure must not break the context manager.
                with session_guard("s007"):
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
