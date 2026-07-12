"""Tests for voice profile helper integration with the consolidation cycle.

Verifies the document-session eviction predicate governs all helper
invocations: any document session in the pending batch evicts the GPU voice
pair for the duration of the cycle (it is the unbounded-density extraction
regime that exhausts the 8 GiB device); a pure-transcript batch keeps the
GPU voice pair resident.

Also pins the GPU-lock topology of the shared extraction stage
(``_extract_pending_sessions``), which both consolidation paths call:

- eviction always happens INSIDE the lock (``lock_held=True``);
- the interim tick passes ``lock_held=False`` — the stage acquires the lock
  itself and releases it on return, so every interim restore (no-facts,
  simulate, replay-disabled, and the ExtractionFailed abort) runs OUTSIDE the
  lock (``lock_held=False``); the post-training restore inside the BG worker
  is the one exception (``lock_held=True`` — the worker holds the lock);
- the full cycle's consume-pending pre-stage passes ``lock_held=True`` — it
  already runs under the BackgroundTrainer worker's lock, so the stage must
  NOT acquire (the lock is non-reentrant: a second acquisition deadlocks the
  worker with ``_state["consolidating"]`` stuck True) and the restore also
  runs inside the lock (``lock_held=True``);
- the stage RETURNS its ``ExtractionFailed`` abort instead of raising, so both
  callers can still see whether voice was evicted and restore it with the
  ``lock_held`` value their own lock context demands.
"""

from __future__ import annotations

import threading
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


def _make_state_patch(pending, config, target_profile="gpu"):
    """Build the ``_state`` patch + mock SessionBuffer shared by these tests.

    Returns ``(state_patch, mock_buffer)``.  Used by both the whole-cycle
    patcher (``_patch_extract_training``) and the extraction-stage patcher
    (``_patch_extraction_stage``) so both exercise the same fixture shape.
    """
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
    return state_patch, mock_buffer


@contextmanager
def _patch_extract_training(pending, config, target_profile="gpu"):
    """Patch _state for _extract_and_start_training calls."""
    import paramem.server.app as app_module

    state_patch, mock_buffer = _make_state_patch(pending, config, target_profile)

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


@contextmanager
def _patch_extraction_stage(pending, config, target_profile="gpu", *, real_lock=False):
    """Patch _state for direct ``_extract_pending_sessions`` calls.

    Args:
        real_lock: When ``True`` the REAL ``gpu_lock_sync`` is left in place so
            a test can prove the stage does not re-acquire a lock the caller
            already holds.  Otherwise the lock is a no-op mock (as in the
            whole-cycle patcher).
    """
    import paramem.server.app as app_module

    state_patch, mock_buffer = _make_state_patch(pending, config, target_profile)

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._set_voice_pipeline_profile") as mock_profile,
        patch("paramem.server.app.check_vram_headroom"),
        patch("paramem.server.app.vram_scope") as mock_vram_scope,
    ):
        mock_vram_scope.return_value.__enter__ = MagicMock(return_value=None)
        mock_vram_scope.return_value.__exit__ = MagicMock(return_value=False)
        if real_lock:
            yield mock_profile, mock_buffer
        else:
            with patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock:
                mock_lock.return_value.__enter__ = MagicMock(return_value=None)
                mock_lock.return_value.__exit__ = MagicMock(return_value=False)
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


def test_interim_extraction_failed_abort_restores_voice_outside_lock(tmp_path):
    """Interim path: the ExtractionFailed abort restores voice with ``lock_held=False``.

    The extraction stage owns the lock on this path: ``_extract_and_start_training``
    calls it with ``lock_held=False``, so the stage acquires ``gpu_lock_sync()``,
    evicts voice inside it (``lock_held=True``), and RETURNS the abort — releasing
    the lock on the way out.  The caller's restore therefore runs OUTSIDE the lock
    and must pass ``lock_held=False``; passing ``True`` there would tell the voice
    helper a lock is held when it is not, and the eventual release would be unpaired.

    The abort is *returned*, never raised: a raise would leave the caller no way to
    learn that voice was evicted, and a ``finally``-based restore would fire with the
    GPU lock still held (deadlock — ``_state["consolidating"]`` stuck True, every
    subsequent /consolidate answered "deferred_already_running", and the retry path
    mandated by the extraction-failure-fails-cycle policy silently blocked).

    The matching evict call (cpu, lock_held=True) confirms the lock context: the
    eviction is inside the with-block, the restore is after it.
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
    # Eviction — inside the stage's gpu_lock_sync block — must be lock_held=True.
    assert calls[0] == call("cpu", lock_held=True), f"Expected cpu eviction, got {calls[0]}"
    # Abort-path restore — the stage released the lock when it returned, so the
    # caller restores outside it.
    assert calls[1] == call("gpu", lock_held=False), (
        f"Expected gpu restore with lock_held=False (the stage released the GPU "
        f"lock when it returned the abort), got {calls[1]}"
    )
    # Nothing may retire on an abort.
    mock_buffer.mark_consolidated.assert_not_called()
    # And the flag must clear so the next /consolidate isn't deferred.
    assert app_module._state["consolidating"] is False, (
        "ExtractionFailed abort must clear consolidating; otherwise the retry "
        "path mandated by the extraction-failure-fails-cycle policy is blocked."
    )


# ---------------------------------------------------------------------------
# Tests — the shared extraction stage (_extract_pending_sessions)
# ---------------------------------------------------------------------------


def test_extraction_stage_returns_abort_and_never_raises(tmp_path):
    """``ExtractionFailed`` is RETURNED in ``aborted``, not raised.

    The whole lock topology depends on this: on a raise path there is no return
    value, so neither caller could learn whether voice was evicted, and the
    obvious ``finally: if evicted`` restore would run with the GPU lock still
    held (consume-pending) or raise ``UnboundLocalError`` from inside ``finally``,
    masking the real exception and killing the worker with the lock held.
    """
    import paramem.server.app as app_module
    from paramem.graph.extractor import ExtractionFailed

    pending = _make_pending(source_type="document", n=2)
    config = _make_config(tmp_path=tmp_path)

    with _patch_extraction_stage(pending, config) as (mock_profile, mock_buffer):
        loop = _make_loop_no_qa()
        loop.extract_session.side_effect = ExtractionFailed("sota_enrich", "unparseable")

        # No raise — the abort comes back on the result object.
        result = app_module._extract_pending_sessions(loop, lock_held=False)

    assert isinstance(result.aborted, ExtractionFailed)
    assert result.aborted.phase == "sota_enrich"
    assert result.evicted_voice is True, "caller needs this bit to restore voice on abort"
    # Abort on the FIRST chunk stops the batch — the second is never attempted.
    assert loop.extract_session.call_count == 1
    # The stage evicts, and leaves the restore to the caller.
    assert mock_profile.call_args_list == [call("cpu", lock_held=True)]


def test_extraction_stage_does_not_reacquire_a_held_lock(tmp_path):
    """``lock_held=True`` must NOT acquire ``gpu_lock_sync`` — deadlock proof (CPU).

    The consume-pending pre-stage runs on the BackgroundTrainer worker thread,
    which already holds the non-reentrant ``_gpu_thread_lock`` for the whole
    fold.  Here the REAL lock is held while the stage runs on another thread
    with ``lock_held=True``: if the stage acquired the lock it would block
    forever (in production: the worker deadlocks against itself, the fold never
    finishes, ``_state["consolidating"]`` stays True and the server is hung).
    The watchdog join turns that hang into a failing assertion.
    """
    import paramem.server.app as app_module
    from paramem.server.gpu_lock import gpu_lock_sync

    pending = _make_pending(source_type="document", n=1)
    config = _make_config(tmp_path=tmp_path)
    box: dict = {}

    with _patch_extraction_stage(pending, config, real_lock=True) as (mock_profile, mock_buffer):
        loop = _make_loop_no_qa()

        def _run() -> None:
            box["result"] = app_module._extract_pending_sessions(loop, lock_held=True)

        with gpu_lock_sync():  # the worker's lock, held for the whole call
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            worker.join(timeout=10)
            # Capture liveness WHILE the real lock is still held, not after the
            # `with` exits. On a regression (the stage re-acquires the lock
            # unconditionally) the worker is genuinely still blocked at this
            # point; releasing the lock first would let an unblocked worker
            # finish before the assertion runs on an unlucky GIL switch, so
            # the proof must not depend on scheduling timing.
            still_blocked = worker.is_alive()

    assert not still_blocked, (
        "_extract_pending_sessions(lock_held=True) blocked on gpu_lock_sync — it must "
        "not acquire a lock the caller already holds (non-reentrant → deadlocked worker)"
    )
    assert box["result"].session_ids == ["sid-0"]
    assert box["result"].aborted is None
    # Eviction still happens, and still declares the lock as held.
    assert mock_profile.call_args_list == [call("cpu", lock_held=True)]


def test_extraction_stage_failed_chunk_stays_pending_survivor_retires(tmp_path):
    """A VramExhausted chunk stays pending; the chunk that succeeded may retire."""
    import paramem.server.app as app_module
    from paramem.server.vram_guard import VramExhausted

    pending = _make_pending(source_type="document", n=2)
    config = _make_config(tmp_path=tmp_path)

    with _patch_extraction_stage(pending, config) as (mock_profile, mock_buffer):
        # retirable() is the document-atomic gate; pass sessions through so the
        # assertion isolates the extraction-failure filter.
        mock_buffer.retirable.side_effect = lambda ids: sorted(ids)

        loop = _make_loop_no_qa()

        def _extract(_transcript, session_id, **_kwargs):
            if session_id == "sid-0":
                raise VramExhausted("extract")
            return ([{"subject": "s"}], [])

        loop.extract_session.side_effect = _extract

        result = app_module._extract_pending_sessions(loop, lock_held=False)

        # Read the OOM record while the patched _state is still installed.
        assert app_module._state["chunk_failures"][0]["session_id"] == "sid-0"

    assert loop.extract_session.call_count == 2, "an OOM chunk must not poison the next one"
    assert result.aborted is None
    assert result.session_ids == ["sid-0", "sid-1"]
    assert result.failed_session_ids == {"sid-0"}
    assert result.completed_session_ids(mock_buffer) == ["sid-1"]
    # Provenance stamping survives the dedup.
    assert result.episodic_rels == [{"subject": "s", "speaker_id": "spk-1", "session_id": "sid-1"}]


# ---------------------------------------------------------------------------
# Tests — the full cycle's consume-pending pre-stage (max_interim_count == 0)
# ---------------------------------------------------------------------------


def _run_consume_pending_cycle(config, loop):
    """Drive the full cycle's Stage-B body directly and return (outcome, profile calls).

    ``_run_full_consolidation_sync`` hands its body to ``_run_stage_b_cycle``,
    which submits it to the BackgroundTrainer worker (the thread that holds the
    GPU lock).  These tests capture that body and invoke it themselves, which is
    exactly the lock context production gives it — the caller holds the lock,
    so the pre-stage must run with ``lock_held=True`` throughout.
    """
    import paramem.server.app as app_module

    captured: dict = {}

    def _capture(*, kind, incident_key, failure_summary, failure_detail, body):
        captured["body"] = body

    pending = _make_pending(source_type="document", n=1)
    with _patch_extraction_stage(pending, config) as (mock_profile, mock_buffer):
        with patch("paramem.server.app._run_stage_b_cycle", side_effect=_capture):
            app_module._run_full_consolidation_sync()
        outcome, _finalizer = captured["body"](loop, MagicMock())

    return outcome, mock_profile.call_args_list, mock_buffer


def test_consume_pending_evicts_and_restores_voice_inside_worker_lock(tmp_path):
    """Consume-pending: BOTH the eviction and the restore run with ``lock_held=True``.

    The pre-stage runs inside the BG worker's GPU lock, which it must neither
    re-acquire (stage: ``lock_held=True``) nor pretend to be free at restore time.
    """
    config = _make_config(tmp_path=tmp_path)
    config.consolidation.max_interim_count = 0  # consume-pending mode
    config.consolidation.mode = "train"

    loop = _make_loop_no_qa()
    loop.consolidate.return_value = {"status": "complete", "tiers_rebuilt": []}

    outcome, profile_calls, mock_buffer = _run_consume_pending_cycle(config, loop)

    assert outcome == "noop"
    assert profile_calls == [
        call("cpu", lock_held=True),  # eviction, inside the worker's lock
        call("gpu", lock_held=True),  # restore, still inside the worker's lock
    ]
    # Extraction succeeded (no facts after dedup) → the session retires.
    mock_buffer.mark_consolidated.assert_called_once()


def test_consume_pending_abort_restores_voice_and_returns_extraction_failed(tmp_path):
    """Consume-pending: an ExtractionFailed abort restores voice (lock_held=True) and
    terminates the fold without retiring anything."""
    from paramem.graph.extractor import ExtractionFailed

    config = _make_config(tmp_path=tmp_path)
    config.consolidation.max_interim_count = 0
    config.consolidation.mode = "train"

    loop = _make_loop_no_qa()
    loop.extract_session.side_effect = ExtractionFailed("sota_enrich", "unparseable")

    outcome, profile_calls, mock_buffer = _run_consume_pending_cycle(config, loop)

    assert outcome == "extraction_failed"
    assert profile_calls == [
        call("cpu", lock_held=True),
        call("gpu", lock_held=True),
    ]
    # The fold must not run after an aborted pre-stage, and nothing may retire.
    loop.consolidate.assert_not_called()
    mock_buffer.mark_consolidated.assert_not_called()


def test_standard_full_cycle_runs_no_extraction_and_no_voice_swap(tmp_path):
    """max_interim_count > 0: no pre-stage, no extraction, no voice eviction.

    The standard full cycle folds already-trained interim slots into main — it
    never runs the extraction chain, so the KV-cache spikes that motivate
    eviction do not apply.
    """
    config = _make_config(tmp_path=tmp_path)
    config.consolidation.max_interim_count = 7
    config.consolidation.mode = "train"

    loop = _make_loop_no_qa()
    loop.consolidate.return_value = {"status": "complete", "tiers_rebuilt": []}

    outcome, profile_calls, mock_buffer = _run_consume_pending_cycle(config, loop)

    assert outcome == "noop"
    assert profile_calls == []
    loop.extract_session.assert_not_called()
    mock_buffer.mark_consolidated.assert_not_called()


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
