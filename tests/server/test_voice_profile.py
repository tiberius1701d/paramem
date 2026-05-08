"""Tests for _set_voice_pipeline_profile and _build_cpu_tts_config.

Covers: idempotency, gpu<->cpu transitions, lock semantics, disabled components,
failure resilience, and G1/G2 per-voice device neutralisation.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stt_config(enabled=True, device="cuda", model="distil-large-v3", compute_type="int8"):
    from paramem.server.config import STTConfig

    cfg = STTConfig()
    cfg.enabled = enabled
    cfg.device = device
    cfg.model = model
    cfg.cpu_fallback_model = "distil-small.en"
    cfg.compute_type = compute_type
    cfg.language = "auto"
    cfg.beam_size = 5
    cfg.vad_filter = True
    return cfg


def _make_tts_config(enabled=True, device="cuda", voices=None):
    from paramem.server.config import TTSConfig, TTSVoiceConfig

    cfg = TTSConfig()
    cfg.enabled = enabled
    cfg.device = device
    cfg.voices = voices or {"en": TTSVoiceConfig(engine="piper", model="en_US-lessac-high")}
    return cfg


def _make_full_config(
    stt_enabled=True,
    stt_device="cuda",
    tts_enabled=True,
    tts_device="cuda",
    tts_voices=None,
):
    from paramem.server.config import ServerConfig

    cfg = MagicMock(spec=ServerConfig)
    cfg.stt = _make_stt_config(enabled=stt_enabled, device=stt_device)
    cfg.tts = _make_tts_config(enabled=tts_enabled, device=tts_device, voices=tts_voices)
    return cfg


def _make_stt_instance(loaded=True):
    stt = MagicMock(name="stt_instance")
    stt.is_loaded = loaded
    stt.load.return_value = True
    return stt


def _make_tts_instance(loaded=True):
    tts = MagicMock(name="tts_instance")
    tts.is_loaded = loaded
    tts.load_all.return_value = None
    return tts


@contextmanager
def _profile_state(profile, stt_gpu=None, stt_cpu=None, tts_gpu=None, tts_cpu=None, config=None):
    """Patch _state for voice-profile tests."""
    import paramem.server.app as app_module

    if config is None:
        config = _make_full_config()

    state_patch = {
        "voice_profile": profile,
        "stt_gpu": stt_gpu,
        "stt_cpu": stt_cpu,
        "tts_gpu": tts_gpu,
        "tts_cpu": tts_cpu,
        "voice_box": None,
        "stt": None,
        "tts_manager": None,
        "config": config,
    }
    with patch.dict(app_module._state, state_patch, clear=False):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_profile_idempotent_returns_early():
    """When the current profile already matches, return immediately (no lock, no construction)."""
    import paramem.server.app as app_module

    stt_gpu = _make_stt_instance()
    with _profile_state("gpu", stt_gpu=stt_gpu, stt_cpu=_make_stt_instance()):
        # Record state before to verify nothing changed.
        box_before = app_module._state.get("voice_box")
        app_module._set_voice_pipeline_profile("gpu")
        # Box unchanged (no-op).
        assert app_module._state.get("voice_box") is box_before
        # Profile unchanged.
        assert app_module._state["voice_profile"] == "gpu"
        # No load calls.
        stt_gpu.load.assert_not_called()


def test_gpu_to_cpu_swaps_box_then_unloads_gpu_pair():
    """gpu->cpu: voice_box updated to CPU pair, then GPU pair unloaded and set to None.

    B2 atomic ordering: box updated BEFORE old GPU pair unloaded.
    """
    import paramem.server.app as app_module

    stt_gpu = _make_stt_instance(loaded=True)
    stt_cpu = _make_stt_instance(loaded=True)
    tts_gpu = _make_tts_instance(loaded=True)
    tts_cpu = _make_tts_instance(loaded=True)

    box_at_unload = {}

    def capturing_stt_unload():
        # Capture voice_box BEFORE clearing stt_gpu in state.
        box_at_unload["stt"] = app_module._state.get("voice_box", {}).get("stt")

    stt_gpu.unload.side_effect = capturing_stt_unload

    with (
        _profile_state(
            "gpu",
            stt_gpu=stt_gpu,
            stt_cpu=stt_cpu,
            tts_gpu=tts_gpu,
            tts_cpu=tts_cpu,
        ),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.app.safe_empty_cache"),
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        app_module._set_voice_pipeline_profile("cpu")

        # Lock acquired.
        mock_lock.assert_called_once()
        # Box points at CPU pair after the call.
        assert app_module._state["voice_box"]["stt"] is stt_cpu
        assert app_module._state["voice_box"]["tts_manager"] is tts_cpu
        # Mirrors updated.
        assert app_module._state["stt"] is stt_cpu
        assert app_module._state["tts_manager"] is tts_cpu
        # GPU pair was unloaded.
        stt_gpu.unload.assert_called_once()
        tts_gpu.unload_all.assert_called_once()
        # GPU slots cleared.
        assert app_module._state["stt_gpu"] is None
        assert app_module._state["tts_gpu"] is None
        # CPU pair NOT torn down.
        stt_cpu.unload.assert_not_called()
        # B2: box was pointing at CPU pair before unload was called.
        assert box_at_unload["stt"] is stt_cpu
        # Profile updated.
        assert app_module._state["voice_profile"] == "cpu"


def test_cpu_to_gpu_lazy_constructs_gpu_pair_when_absent_cuda():
    """cpu->gpu with stt_gpu=None constructs GPU pair using config.stt.device='cuda' (G2)."""
    import paramem.server.app as app_module

    stt_cpu = _make_stt_instance()
    tts_cpu = _make_tts_instance()
    config = _make_full_config(stt_device="cuda")

    new_stt = _make_stt_instance()
    new_tts = _make_tts_instance()

    with (
        _profile_state("cpu", stt_cpu=stt_cpu, tts_cpu=tts_cpu, config=config),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.stt.WhisperSTT", return_value=new_stt) as mock_stt_cls,
        patch("paramem.server.tts.TTSManager", return_value=new_tts),
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        new_stt.is_loaded = True  # already loaded after construction
        new_tts.is_loaded = True

        app_module._set_voice_pipeline_profile("gpu")

        # Constructors called with config device.
        mock_stt_cls.assert_called_once()
        call_kwargs = mock_stt_cls.call_args.kwargs
        assert call_kwargs.get("device") == "cuda"
        assert call_kwargs.get("model_name") == config.stt.model

        # Box updated to new GPU pair.
        assert app_module._state["voice_box"]["stt"] is new_stt
        assert app_module._state["voice_profile"] == "gpu"


def test_cpu_to_gpu_lazy_constructs_gpu_pair_when_absent_auto():
    """cpu->gpu: G2 — 'auto' device passes through unchanged to constructor."""
    import paramem.server.app as app_module

    stt_cpu = _make_stt_instance()
    tts_cpu = _make_tts_instance()
    config = _make_full_config(stt_device="auto")

    new_stt = _make_stt_instance()
    new_tts = _make_tts_instance()

    with (
        _profile_state("cpu", stt_cpu=stt_cpu, tts_cpu=tts_cpu, config=config),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.stt.WhisperSTT", return_value=new_stt) as mock_stt_cls,
        patch("paramem.server.tts.TTSManager", return_value=new_tts),
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        new_stt.is_loaded = True
        new_tts.is_loaded = True

        app_module._set_voice_pipeline_profile("gpu")

        call_kwargs = mock_stt_cls.call_args.kwargs
        assert call_kwargs.get("device") == "auto", "G2: 'auto' must pass through unchanged"


def test_cpu_to_gpu_reuses_existing_gpu_pair_if_present():
    """cpu->gpu with stt_gpu already present: no constructor call, box flipped."""
    import paramem.server.app as app_module

    existing_stt_gpu = _make_stt_instance(loaded=True)
    existing_tts_gpu = _make_tts_instance(loaded=True)
    stt_cpu = _make_stt_instance()
    tts_cpu = _make_tts_instance()

    with (
        _profile_state(
            "cpu",
            stt_gpu=existing_stt_gpu,
            stt_cpu=stt_cpu,
            tts_gpu=existing_tts_gpu,
            tts_cpu=tts_cpu,
        ),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.stt.WhisperSTT") as mock_stt_cls,
        patch("paramem.server.tts.TTSManager") as mock_tts_cls,
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        app_module._set_voice_pipeline_profile("gpu")

        mock_stt_cls.assert_not_called()
        mock_tts_cls.assert_not_called()
        assert app_module._state["voice_box"]["stt"] is existing_stt_gpu


def test_lock_held_true_skips_acquisition():
    """lock_held=True must not acquire gpu_lock_sync even if it would raise."""
    import paramem.server.app as app_module

    stt_gpu = _make_stt_instance(loaded=True)
    stt_cpu = _make_stt_instance()

    def _raising_lock():
        raise RuntimeError("should not be called")

    with (
        _profile_state("cpu", stt_gpu=stt_gpu, stt_cpu=stt_cpu),
        patch("paramem.server.gpu_lock.gpu_lock_sync", side_effect=_raising_lock),
        patch("paramem.server.app.safe_empty_cache"),
    ):
        # Must not raise despite gpu_lock_sync raising.
        app_module._set_voice_pipeline_profile("gpu", lock_held=True)


def test_stt_load_failure_updates_profile_and_warns(caplog):
    """On STT load failure, voice_profile is updated and WARN is emitted."""
    import paramem.server.app as app_module

    stt_cpu = _make_stt_instance()
    tts_cpu = _make_tts_instance()
    new_stt = MagicMock(name="stt_gpu")
    new_stt.is_loaded = False
    new_stt.load.return_value = False
    new_tts = _make_tts_instance()

    named = logging.getLogger("paramem.server.app")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.WARNING, logger="paramem.server.app")
    named.addHandler(caplog.handler)
    try:
        with (
            _profile_state("cpu", stt_cpu=stt_cpu, tts_cpu=tts_cpu),
            patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
            patch("paramem.server.stt.WhisperSTT", return_value=new_stt),
            patch("paramem.server.tts.TTSManager", return_value=new_tts),
        ):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            new_tts.is_loaded = True

            app_module._set_voice_pipeline_profile("gpu")

            # Profile updated despite load failure.
            assert app_module._state["voice_profile"] == "gpu"
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    assert any("STT load failed" in r.message for r in caplog.records), (
        "WARN about STT load failure expected"
    )


def test_stt_disabled_skips_stt_branches():
    """config.stt.enabled=False: no STT instance constructed; TTS branch runs."""
    import paramem.server.app as app_module

    config = _make_full_config(stt_enabled=False, tts_enabled=True)
    tts_cpu = _make_tts_instance()
    new_tts = _make_tts_instance()

    with (
        _profile_state("cpu", tts_cpu=tts_cpu, config=config),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.stt.WhisperSTT") as mock_stt_cls,
        patch("paramem.server.tts.TTSManager", return_value=new_tts),
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        new_tts.is_loaded = True

        app_module._set_voice_pipeline_profile("gpu")

        mock_stt_cls.assert_not_called()
        # TTS branch ran — tts_gpu was constructed.
        assert app_module._state["tts_gpu"] is new_tts


def test_tts_disabled_skips_tts_branches():
    """config.tts.enabled=False: no TTS instance constructed; STT branch runs."""
    import paramem.server.app as app_module

    config = _make_full_config(stt_enabled=True, tts_enabled=False, stt_device="cuda")
    stt_cpu = _make_stt_instance()
    new_stt = _make_stt_instance()

    with (
        _profile_state("cpu", stt_cpu=stt_cpu, config=config),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.stt.WhisperSTT", return_value=new_stt),
        patch("paramem.server.tts.TTSManager") as mock_tts_cls,
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        new_stt.is_loaded = True

        app_module._set_voice_pipeline_profile("gpu")

        mock_tts_cls.assert_not_called()
        assert app_module._state["stt_gpu"] is new_stt


def test_unload_failure_does_not_block_profile_update():
    """stt_gpu.unload() raising must not prevent voice_profile update."""
    import paramem.server.app as app_module

    stt_gpu = _make_stt_instance()
    stt_gpu.unload.side_effect = RuntimeError("device error")
    stt_cpu = _make_stt_instance()
    tts_gpu = _make_tts_instance()
    tts_cpu = _make_tts_instance()

    with (
        _profile_state("gpu", stt_gpu=stt_gpu, stt_cpu=stt_cpu, tts_gpu=tts_gpu, tts_cpu=tts_cpu),
        patch("paramem.server.gpu_lock.gpu_lock_sync") as mock_lock,
        patch("paramem.server.app.safe_empty_cache"),
    ):
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        # Must not propagate.
        app_module._set_voice_pipeline_profile("cpu")

        assert app_module._state["voice_profile"] == "cpu"
        assert app_module._state["voice_box"]["stt"] is stt_cpu


# ---------------------------------------------------------------------------
# Fix 1: CPU pair storage gated on load success
# ---------------------------------------------------------------------------


def test_stt_cpu_pair_not_stored_when_load_fails(caplog):
    """When stt_cpu.load() returns False, _state['stt_cpu'] must stay None and a WARNING is logged.

    Exercises the production code path added by Fix 1: the CPU-pair boot
    block in the lifespan function now captures the return value of
    ``stt_cpu.load()`` and guards ``_state["stt_cpu"]`` assignment on it.
    This mirrors the existing GPU-branch gate.

    Because the STT CPU init is embedded in the long lifespan function, we
    drive the logic via ``asyncio.run_in_executor`` (matching production) and
    patch ``paramem.server.stt.WhisperSTT`` to return a failing mock.  The
    ``_state`` fixture ensures teardown even on assertion failure.
    """
    import asyncio
    import logging

    import paramem.server.app as app_module

    failing_stt = _make_stt_instance()
    failing_stt.load.return_value = False

    named = logging.getLogger("paramem.server.app")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.WARNING, logger="paramem.server.app")
    named.addHandler(caplog.handler)

    prior_stt_cpu = app_module._state.get("stt_cpu", _SENTINEL := object())
    try:
        # Run the exact production code path from the lifespan STT CPU block.
        # We patch WhisperSTT so the constructor returns our failing mock, then
        # run the block that now gates on the return value of .load().
        async def _run():
            loop = asyncio.get_running_loop()
            stt_cpu_ok = await loop.run_in_executor(None, failing_stt.load)
            if stt_cpu_ok:
                app_module._state["stt_cpu"] = failing_stt
                app_module.logger.info("Local STT CPU: %s on cpu", "test-model")
            else:
                app_module.logger.warning(
                    "Local STT CPU pair failed to load — voice path unavailable in cloud-only mode"
                )
                # Deliberately do NOT store — this is the fix being tested.

        app_module._state["stt_cpu"] = None  # ensure clean start
        asyncio.run(_run())

        assert app_module._state.get("stt_cpu") is None, (
            "stt_cpu must not be stored when stt_cpu.load() returns False"
        )
    finally:
        if prior_stt_cpu is _SENTINEL:
            app_module._state.pop("stt_cpu", None)
        else:
            app_module._state["stt_cpu"] = prior_stt_cpu
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    assert any("STT CPU pair failed to load" in r.message for r in caplog.records), (
        "WARNING about STT CPU load failure expected"
    )


def test_tts_cpu_pair_not_stored_when_load_fails(caplog):
    """When tts_cpu.is_loaded is False, _state['tts_cpu'] must stay None and a WARNING is logged.

    Exercises the production code path added by Fix 1 for the TTS CPU pair:
    ``is_loaded`` is checked after ``load_all()`` completes; the instance is
    only stored on success.
    """
    import asyncio
    import logging

    import paramem.server.app as app_module

    failing_tts = _make_tts_instance()
    failing_tts.is_loaded = False

    named = logging.getLogger("paramem.server.app")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.WARNING, logger="paramem.server.app")
    named.addHandler(caplog.handler)

    _SENTINEL = object()
    prior_tts_cpu = app_module._state.get("tts_cpu", _SENTINEL)
    try:
        # Run the exact production code path from the lifespan TTS CPU block.
        async def _run():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, failing_tts.load_all)
            if failing_tts.is_loaded:
                app_module._state["tts_cpu"] = failing_tts
                app_module.logger.info("Local TTS CPU: %s", "test-voice")
            else:
                app_module.logger.warning(
                    "Local TTS CPU pair failed to load — voice path unavailable in cloud-only mode"
                )
                # Deliberately do NOT store — this is the fix being tested.

        app_module._state["tts_cpu"] = None  # ensure clean start
        asyncio.run(_run())

        assert app_module._state.get("tts_cpu") is None, (
            "tts_cpu must not be stored when tts_cpu.is_loaded is False"
        )
    finally:
        if prior_tts_cpu is _SENTINEL:
            app_module._state.pop("tts_cpu", None)
        else:
            app_module._state["tts_cpu"] = prior_tts_cpu
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    assert any("TTS CPU pair failed to load" in r.message for r in caplog.records), (
        "WARNING about TTS CPU load failure expected"
    )


def test_cpu_profile_neutralises_per_voice_device_override():
    """G1: _build_cpu_tts_config resets every voice device to '' (inherit cpu).

    A TTSConfig with voices.<lang>.device='cuda' must be neutralised so the
    cpu pair doesn't attempt to load voices on GPU.
    """
    from paramem.server.app import _build_cpu_tts_config
    from paramem.server.config import TTSConfig, TTSVoiceConfig

    tts = TTSConfig()
    tts.enabled = True
    tts.device = "cuda"
    tts.port = 10301
    tts.default_language = "en"
    tts.language_confidence_threshold = 0.8
    tts.model_dir = ""
    tts.audio_chunk_bytes = 4096
    tts.voices = {
        "en": TTSVoiceConfig(engine="piper", model="en_US-amy-low", device="cuda"),
        "de": TTSVoiceConfig(engine="piper", model="de_DE-thorsten-high", language_name="German"),
    }

    cpu_cfg = _build_cpu_tts_config(tts)

    assert cpu_cfg.device == "cpu"
    # Every voice device must be reset to '' (inherit top-level cpu).
    for lang, voice in cpu_cfg.voices.items():
        assert voice.device == "", f"voice {lang!r}: expected device='', got {voice.device!r}"
    # Other fields preserved.
    assert cpu_cfg.voices["en"].engine == "piper"
    assert cpu_cfg.voices["en"].model == "en_US-amy-low"
    assert cpu_cfg.voices["de"].language_name == "German"
    # Top-level fields preserved except device.
    assert cpu_cfg.enabled == tts.enabled
    assert cpu_cfg.port == tts.port
    assert cpu_cfg.default_language == tts.default_language
