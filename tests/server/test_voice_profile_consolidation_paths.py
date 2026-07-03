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
            "doc_title": None,
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:05:00+00:00",
        }
        for i in range(n)
    ]


def _make_loop_no_qa():
    """Loop that returns empty QA so the no-facts early exit fires."""
    loop = MagicMock()
    loop.shutdown_requested = False
    loop.config.indexed_key_replay = True
    loop.config.consolidation.mode = "train"
    # extract_session returns empty lists — no QA extracted.
    loop.extract_session.return_value = ([], [])
    return loop


def _make_config(mode="train", tmp_path=None):
    """Build a minimal mock config for voice-profile consolidation tests.

    Args:
        mode: ``config.consolidation.mode`` value.
        tmp_path: When provided, set ``config.paths.data`` to this real path so
            that incident/run-status I/O writes (``record_last_run``,
            ``record_incident``) land in ``tmp_path / "state"`` rather than
            creating a literal ``MagicMock/`` directory at the repo root.
            All tests that call ``_extract_and_start_training`` must supply this.
    """
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
    # Ground incident/run-status I/O in a real path when provided.
    if tmp_path is not None:
        cfg.paths.data = tmp_path
    return cfg


@contextmanager
def _patch_extract_training(pending, config, target_profile="gpu"):
    """Patch _state for _extract_and_start_training calls."""
    import paramem.server.app as app_module

    mock_buffer = MagicMock()
    mock_buffer.get_pending.return_value = pending
    mock_buffer.pending_count = len(pending)
    mock_buffer.mark_consolidated.return_value = None
    # The NAMED-only filter at extraction time (app.py) reads pending_facts()
    # and keeps only sessions that classify_session() rules NAMED. Mirror
    # get_pending() exactly: every session carries a real speaker_id and the
    # fixture's speaker_store is None (so _is_anon_ex is False) → all NAMED →
    # the filtered `pending` equals get_pending(), matching the pre-filter
    # behavior these tests assert.
    mock_buffer.pending_facts.return_value = [
        {
            "session_id": s["session_id"],
            "speaker_id": s["speaker_id"],
            "has_voice_embedding": False,
            "age_seconds": 0,
        }
        for s in pending
    ]

    from paramem.memory.store import MemoryStore as _MS

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
        patch("paramem.server.app.create_consolidation_loop"),
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
# Tests
# ---------------------------------------------------------------------------


def test_document_cycle_swaps_to_cpu_then_gpu_in_local_mode(tmp_path):
    """Document session present: helper called with cpu (lock_held=True) then gpu (lock_held=False).

    The ``evict_voice_for_cycle`` predicate gates both calls.
    Lock-acquisition matrix: eviction call is inside gpu_lock_sync()
    (lock_held=True); no-facts restore is outside the with-block (lock_held=False).
    """
    import paramem.server.app as app_module

    pending = _make_pending(source_type="document", n=1)
    config = _make_config(tmp_path=tmp_path)

    with _patch_extract_training(pending, config, target_profile="gpu") as (
        mock_profile,
        mock_buffer,
    ):
        # Loop returns empty QA so we hit the no-facts early exit at line 6636.
        loop = _make_loop_no_qa()
        with patch("paramem.server.app.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    # First call: evict to cpu with lock_held=True.
    assert calls[0] == call("cpu", lock_held=True), f"Expected cpu eviction, got {calls[0]}"
    # Second call: restore to gpu with lock_held=False (no-facts path, outside lock scope).
    assert calls[1] == call("gpu", lock_held=False), f"Expected gpu restore, got {calls[1]}"


def test_voice_cycle_in_cloud_only_targets_cpu_after(tmp_path):
    """In cloud-only mode, the post-cycle restore targets cpu via _target_profile()."""
    import paramem.server.app as app_module

    pending = _make_pending(source_type="document", n=1)
    config = _make_config(tmp_path=tmp_path)

    with _patch_extract_training(pending, config, target_profile="cpu") as (
        mock_profile,
        mock_buffer,
    ):
        # Loop returns empty QA so we hit the no-facts early exit.
        loop = _make_loop_no_qa()
        with patch("paramem.server.app.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    # Both calls: eviction and restore. Post-cycle restore targets cpu (cloud-only).
    assert any(c == call("cpu", lock_held=False) for c in calls), (
        f"Expected cpu restore call in {calls}"
    )


def test_pure_transcript_cycle_skips_voice_swap(tmp_path):
    """A cycle with only transcript sessions must NOT call the helper.

    ``evict_voice_for_cycle`` is False when no session is a document — a
    pure-transcript batch is not in the dense-extraction regime, and the GPU
    voice pair stays resident.
    """
    import paramem.server.app as app_module

    pending = _make_pending(source_type="transcript", n=2)
    config = _make_config(tmp_path=tmp_path)

    with _patch_extract_training(pending, config) as (mock_profile, mock_buffer):
        loop = _make_loop_no_qa()
        with patch("paramem.server.app.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    # No voice profile calls for pure-transcript cycles.
    mock_profile.assert_not_called()


def test_mixed_cycle_with_document_evicts_then_restores_voice(tmp_path):
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
    config = _make_config(tmp_path=tmp_path)

    with _patch_extract_training(pending, config, target_profile="gpu") as (
        mock_profile,
        mock_buffer,
    ):
        loop = _make_loop_no_qa()
        with patch("paramem.server.app.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    assert calls[0] == call("cpu", lock_held=True), f"Expected cpu eviction, got {calls[0]}"
    assert calls[1] == call("gpu", lock_held=False), f"Expected gpu restore, got {calls[1]}"


def test_extract_failure_restores_voice_via_done_callback():
    """When _extract_and_start_training raises, _scheduled_extract_done_callback
    calls _set_voice_pipeline_profile(_target_profile(), lock_held=False).

    Note: the restore on failure is implemented in _scheduled_extract_done_callback
    (future callback), not inside _extract_and_start_training itself. The semantics
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


def test_extraction_failed_abort_restores_voice_with_lock_held(tmp_path):
    """ExtractionFailed abort handler must restore voice with ``lock_held=True``.

    Regression for the deadlock discovered 2026-05-17: the abort handler at
    ``_extract_and_start_training`` runs *inside* ``with gpu_lock_sync():``
    (the lock is held from the start of the per-session for-loop).
    ``threading.Lock`` is non-reentrant, so passing ``lock_held=False`` would
    cause the voice restore call to deadlock the executor thread —
    ``_state["consolidating"]`` never clears, the next /consolidate request
    returns "deferred_already_running", and the retry path mandated by
    the extraction-failure-fails-cycle policy is silently blocked.

    The matching evict call at the start of the cycle (cpu, lock_held=True)
    confirms the lock context: both calls happen inside the same with-block.
    """
    import paramem.server.app as app_module
    from paramem.graph.extractor import ExtractionFailed

    pending = _make_pending(source_type="document", n=1)
    config = _make_config(tmp_path=tmp_path)

    with _patch_extract_training(pending, config, target_profile="gpu") as (
        mock_profile,
        mock_buffer,
    ):
        loop = _make_loop_no_qa()
        # Force the abort path: extract_session raises ExtractionFailed.
        loop.extract_session.side_effect = ExtractionFailed(
            "sota_enrich", "cloud enrichment call failed or response unparseable"
        )
        with patch("paramem.server.app.create_consolidation_loop", return_value=loop):
            app_module._extract_and_start_training()

    calls = mock_profile.call_args_list
    # Eviction at cycle start — inside gpu_lock_sync — must be lock_held=True.
    assert calls[0] == call("cpu", lock_held=True), f"Expected cpu eviction, got {calls[0]}"
    # Abort-path restore — STILL inside gpu_lock_sync, so must also be lock_held=True
    # (passing lock_held=False here is the deadlock bug).
    assert calls[1] == call("gpu", lock_held=True), (
        f"Expected gpu restore with lock_held=True (abort path is inside the "
        f"gpu_lock_sync block; lock_held=False would deadlock), got {calls[1]}"
    )
    # And the flag must clear so the next /consolidate isn't deferred.
    assert app_module._state["consolidating"] is False, (
        "ExtractionFailed abort must clear consolidating; otherwise the retry "
        "path mandated by the extraction-failure-fails-cycle policy is blocked."
    )


# ---------------------------------------------------------------------------
# Test — BackgroundTrainer singleton reuse across local cycles
# ---------------------------------------------------------------------------


def test_background_trainer_singleton_reused_across_cycles():
    """The same BackgroundTrainer instance is reused on back-to-back local cycles.

    Exercises _active_bg_trainer directly: calls it twice and asserts:

    1. _build_bg_trainer is called AT MOST ONCE across both _active_bg_trainer calls.
    2. The returned object is the same identity on both calls.
    3. _state["background_trainer"] holds the singleton after the first call.
    4. After the second call (when _state["background_trainer"] is already set),
       bt.model and bt.tokenizer are refreshed from _state.

    Without the singleton fix, each consolidation dispatch site would call
    _build_bg_trainer unconditionally, orphaning the prior worker thread and
    leaking its reserved VRAM (~700 MiB/fold).
    """
    import paramem.server.app as app_module
    from paramem.server.background_trainer import BackgroundTrainer

    config = _make_config()

    construction_count = 0
    sentinel_model_v1 = MagicMock(name="model_v1")
    sentinel_model_v2 = MagicMock(name="model_v2")
    sentinel_tokenizer_v1 = MagicMock(name="tokenizer_v1")
    sentinel_tokenizer_v2 = MagicMock(name="tokenizer_v2")

    sentinel_instance = MagicMock(spec=BackgroundTrainer)
    sentinel_instance.model = sentinel_model_v1
    sentinel_instance.tokenizer = sentinel_tokenizer_v1

    def _tracking_build(cfg):
        nonlocal construction_count
        construction_count += 1
        return sentinel_instance

    state_patch = {
        "config": config,
        "model": sentinel_model_v1,
        "tokenizer": sentinel_tokenizer_v1,
        "background_trainer": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_build_bg_trainer", side_effect=_tracking_build),
    ):
        # First call: singleton does not yet exist — _build_bg_trainer runs once.
        bt_first = app_module._active_bg_trainer(config)
        assert construction_count == 1, (
            f"_build_bg_trainer must be called exactly once on first _active_bg_trainer; "
            f"called {construction_count} time(s)"
        )
        assert bt_first is sentinel_instance
        assert app_module._state["background_trainer"] is sentinel_instance

        # Simulate create_interim_adapter rebinding _state["model"] and _state["tokenizer"]
        # to new wrappers (e.g. after a PeftModel re-wrap or tokenizer reload).
        app_module._state["model"] = sentinel_model_v2
        app_module._state["tokenizer"] = sentinel_tokenizer_v2

        # Second call: singleton already exists — _build_bg_trainer must NOT run again.
        bt_second = app_module._active_bg_trainer(config)

    assert construction_count == 1, (
        f"BackgroundTrainer constructor must be called AT MOST ONCE across two cycles; "
        f"called {construction_count} time(s).  Each extra call orphans the prior worker "
        f"thread and leaks its reserved VRAM."
    )
    assert bt_first is bt_second, (
        f"_active_bg_trainer must return the same object identity on repeated calls; "
        f"got {bt_first!r} then {bt_second!r}"
    )
    # Model handle must have been refreshed to the new wrapper on second call.
    assert sentinel_instance.model is sentinel_model_v2, (
        "_active_bg_trainer must refresh bt.model from _state['model'] on reuse; "
        f"got {sentinel_instance.model!r}"
    )
    # Tokenizer handle must also be refreshed (symmetric to model).
    assert sentinel_instance.tokenizer is sentinel_tokenizer_v2, (
        "_active_bg_trainer must refresh bt.tokenizer from _state['tokenizer'] on reuse; "
        f"got {sentinel_instance.tokenizer!r}"
    )
