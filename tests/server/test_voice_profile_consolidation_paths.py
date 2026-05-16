"""Tests for voice profile helper integration with the consolidation cycle.

Verifies the ``evict_voice_for_cycle`` predicate governs all helper
invocations: any document session in the pending batch evicts the GPU voice
pair for the duration of the cycle (it is the unbounded-density extraction
regime that exhausts the 8 GiB device); a pure-transcript batch keeps the
GPU voice pair resident.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pending(source_type: str = "transcript", n: int = 1) -> list[dict]:
    return [
        {
            "session_id": f"sid-{i}",
            "transcript": "Hello",
            "speaker_id": "spk-1",
            "source_type": source_type,
        }
        for i in range(n)
    ]


def _make_loop_no_qa():
    """Loop that returns empty QA so the no-facts early exit fires."""
    loop = MagicMock()
    loop.shutdown_requested = False
    loop.config.indexed_key_replay_enabled = True
    loop.config.consolidation.mode = "train"
    # extract_session returns empty lists — no QA extracted.
    loop.extract_session.return_value = ([], [])
    return loop


def _make_config(mode="train"):

    cfg = MagicMock()
    cfg.adapters.episodic.enabled = True
    cfg.consolidation.mode = mode
    cfg.consolidation.extraction_stt_correction = False
    cfg.consolidation.extraction_ha_validation = False
    cfg.consolidation.extraction_noise_filter = False
    cfg.consolidation.extraction_noise_filter_model = ""
    cfg.consolidation.extraction_noise_filter_endpoint = ""
    cfg.consolidation.extraction_ner_check = False
    cfg.consolidation.extraction_ner_model = ""
    cfg.consolidation.extraction_plausibility_judge = False
    cfg.consolidation.extraction_plausibility_stage = "post"
    cfg.consolidation.extraction_verify_anonymization = False
    cfg.debug = False
    return cfg


@contextmanager
def _patch_extract_training(pending, config, target_profile="gpu"):
    """Patch _state for _extract_and_start_training calls."""
    import paramem.server.app as app_module

    mock_buffer = MagicMock()
    mock_buffer.get_pending.return_value = pending
    mock_buffer.pending_count = len(pending)
    mock_buffer.mark_consolidated.return_value = None

    from paramem.training.memory_store import MemoryStore as _MS

    state_patch = {
        "config": config,
        "session_buffer": mock_buffer,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "consolidation_loop": None,
        # MemoryStore is lifespan-owned; threaded through to create_consolidation_loop.
        "memory_store": _MS(replay_enabled=False),
        "ha_client": None,
        "speaker_store": None,
        "consolidating": True,
        "mode": "local" if target_profile == "gpu" else "cloud-only",
        "voice_profile": target_profile,
        "chunk_failures": [],
        # router is populated by the lifespan startup in production; the
        # cycle-completion finalize calls router.reload() unconditionally
        # so tests that exercise any completion path need a mock here.
        "router": MagicMock(),
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._set_voice_pipeline_profile") as mock_profile,
        patch("paramem.server.consolidation.create_consolidation_loop"),
        patch("paramem.server.app.assert_free_vram"),
        patch("paramem.server.app.vram_scope") as mock_vram_scope,
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_vram_scope.return_value.__enter__ = MagicMock(return_value=None)
        mock_vram_scope.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_profile, mock_buffer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_document_cycle_swaps_to_cpu_then_gpu_in_local_mode():
    """Document session present: helper called with cpu (lock_held=True) then gpu (lock_held=False).

    The ``evict_voice_for_cycle`` predicate gates both calls.
    Lock-acquisition matrix: eviction call is inside gpu_lock_sync()
    (lock_held=True); no-facts restore is outside the with-block (lock_held=False).
    """
    import paramem.server.app as app_module

    pending = _make_pending(source_type="document", n=1)
    config = _make_config()

    with _patch_extract_training(pending, config, target_profile="gpu") as (
        mock_profile,
        mock_buffer,
    ):
        # Loop returns empty QA so we hit the no-facts early exit at line 6636.
        loop = _make_loop_no_qa()
        with patch("paramem.server.consolidation.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    # First call: evict to cpu with lock_held=True.
    assert calls[0] == call("cpu", lock_held=True), f"Expected cpu eviction, got {calls[0]}"
    # Second call: restore to gpu with lock_held=False (no-facts path, outside lock scope).
    assert calls[1] == call("gpu", lock_held=False), f"Expected gpu restore, got {calls[1]}"


def test_voice_cycle_in_cloud_only_targets_cpu_after():
    """In cloud-only mode, the post-cycle restore targets cpu via _target_profile()."""
    import paramem.server.app as app_module

    pending = _make_pending(source_type="document", n=1)
    config = _make_config()

    with _patch_extract_training(pending, config, target_profile="cpu") as (
        mock_profile,
        mock_buffer,
    ):
        # Loop returns empty QA so we hit the no-facts early exit.
        loop = _make_loop_no_qa()
        with patch("paramem.server.consolidation.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    # Both calls: eviction and restore. Post-cycle restore targets cpu (cloud-only).
    assert any(c == call("cpu", lock_held=False) for c in calls), (
        f"Expected cpu restore call in {calls}"
    )


def test_pure_transcript_cycle_skips_voice_swap():
    """A cycle with only transcript sessions must NOT call the helper.

    ``evict_voice_for_cycle`` is False when no session is a document — a
    pure-transcript batch is not in the dense-extraction regime, and the GPU
    voice pair stays resident.
    """
    import paramem.server.app as app_module

    pending = _make_pending(source_type="transcript", n=2)
    config = _make_config()

    with _patch_extract_training(pending, config) as (mock_profile, mock_buffer):
        loop = _make_loop_no_qa()
        with patch("paramem.server.consolidation.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    # No voice profile calls for pure-transcript cycles.
    mock_profile.assert_not_called()


def test_mixed_cycle_with_document_evicts_then_restores_voice():
    """A cycle mixing a transcript probe + document sessions MUST evict voice.

    This is the regression case: under the prior all()-predicate the lone
    transcript session kept the ~1.5 GiB GPU voice pair resident through the
    dense document extraction, and the plausibility-filter KV-cache growth
    OOM'd mid-generate on the 8 GiB device. ``evict_voice_for_cycle`` is True
    as soon as *any* session is a document, so voice flips to CPU for the
    cycle and back to gpu (local mode) afterwards.
    """
    import paramem.server.app as app_module

    pending = [
        *_make_pending(source_type="transcript", n=1),
        *_make_pending(source_type="document", n=2),
    ]
    config = _make_config()

    with _patch_extract_training(pending, config, target_profile="gpu") as (
        mock_profile,
        mock_buffer,
    ):
        loop = _make_loop_no_qa()
        with patch("paramem.server.consolidation.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    assert calls[0] == call("cpu", lock_held=True), f"Expected cpu eviction, got {calls[0]}"
    assert calls[1] == call("gpu", lock_held=False), f"Expected gpu restore, got {calls[1]}"


def test_extract_failure_restores_voice_via_done_callback():
    """When _extract_and_start_training raises, _scheduled_extract_done_callback
    calls _set_voice_pipeline_profile(_target_profile(), lock_held=False).

    Note: the restore on failure is implemented in _scheduled_extract_done_callback
    (future callback), not inside _extract_and_start_training itself. This is a
    divergence from the plan's W1 note, recorded here as a finding. The semantics
    are equivalent: restore is called with lock_held=False, no lock held at call time.
    """
    import paramem.server.app as app_module

    with (
        patch("paramem.server.app._set_voice_pipeline_profile") as mock_profile,
        patch("paramem.server.app._target_profile", return_value="gpu"),
    ):
        # Simulate a future that raised an exception.
        future = MagicMock()
        exc = RuntimeError("extraction failed")
        future.exception.return_value = exc

        app_module._scheduled_extract_done_callback(future)

    # Restore called with target profile and lock_held=False.
    mock_profile.assert_called_once_with("gpu", lock_held=False)
