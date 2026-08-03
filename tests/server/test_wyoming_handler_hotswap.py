"""Tests for Wyoming handler factory provider-callable indirection.

Verifies that ``start_wyoming_server`` and ``start_wyoming_tts_server``
call the provider callable on every handler factory invocation so that
voice-profile hot-swaps (gpu/cpu) take effect without restarting the
socket listeners.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Wyoming is an optional dependency (`pip install paramem[wyoming]`).
# CI installs only [dev]; skip the whole module when wyoming is absent
# rather than failing at the inner imports below.
pytest.importorskip("wyoming")


def _make_reader_writer():
    """Minimal asyncio stream pair stubs."""
    return MagicMock(name="reader"), MagicMock(name="writer")


async def _call_start_wyoming_server(stt=None, stt_provider=None, speaker_store_provider=None):
    """Create a Wyoming STT server with the given params, return (server, factory)."""
    from unittest.mock import patch

    from paramem.server.wyoming_handler import start_wyoming_server

    captured_factory = {}

    async def _fake_start(factory):
        captured_factory["fn"] = factory

    fake_server = MagicMock()
    fake_server.start = _fake_start

    with patch("paramem.server.wyoming_handler.AsyncServer") as mock_cls:
        mock_cls.from_uri.return_value = fake_server
        server = await start_wyoming_server(
            host="127.0.0.1",
            port=10300,
            stt=stt,
            stt_provider=stt_provider,
            speaker_store_provider=speaker_store_provider,
        )

    return server, captured_factory.get("fn")


async def _call_start_wyoming_tts_server(
    tts_manager=None,
    tts_manager_provider=None,
    speaker_store_provider=None,
):
    """Create a Wyoming TTS server with the given params, return (server, factory)."""
    from unittest.mock import patch

    from paramem.server.wyoming_handler import start_wyoming_tts_server

    captured_factory = {}

    async def _fake_start(factory):
        captured_factory["fn"] = factory

    fake_server = MagicMock()
    fake_server.start = _fake_start

    with patch("paramem.server.wyoming_handler.AsyncServer") as mock_cls:
        mock_cls.from_uri.return_value = fake_server
        server = await start_wyoming_tts_server(
            host="127.0.0.1",
            port=10301,
            tts_manager=tts_manager,
            tts_manager_provider=tts_manager_provider,
            speaker_store_provider=speaker_store_provider,
        )

    return server, captured_factory.get("fn")


def test_stt_provider_called_per_handler_factory_invocation():
    """stt_provider is called on every handler-factory invocation.

    Each Wyoming connection gets the current active STT instance, so a
    gpu->cpu profile swap takes effect for the next connection without
    restarting the socket listener.
    """
    import asyncio

    from paramem.server.wyoming_handler import SpeakerSTTHandler

    stt_v1 = MagicMock(name="stt_gpu")
    stt_v2 = MagicMock(name="stt_cpu")
    call_count = 0
    instances = [stt_v1, stt_v2]

    def provider():
        nonlocal call_count
        result = instances[min(call_count, len(instances) - 1)]
        call_count += 1
        return result

    _server, factory = asyncio.run(_call_start_wyoming_server(stt_provider=provider))
    assert factory is not None, "handler factory must be captured"

    r1, w1 = _make_reader_writer()
    r2, w2 = _make_reader_writer()

    with (
        MagicMock() as _patch1,
        MagicMock() as _patch2,
    ):
        # Call factory twice — provider must be called each time.
        handler1 = factory(r1, w1)
        handler2 = factory(r2, w2)

    assert isinstance(handler1, SpeakerSTTHandler)
    assert isinstance(handler2, SpeakerSTTHandler)
    # provider was called once per factory invocation.
    assert call_count == 2
    # First handler got stt_v1; second got stt_v2.
    assert handler1._stt is stt_v1
    assert handler2._stt is stt_v2


def test_back_compat_stt_param_still_works():
    """When stt_provider is None, the captured ``stt`` param is used (back-compat).

    Existing callers that pass stt= directly must continue to work.
    """
    import asyncio

    from paramem.server.wyoming_handler import SpeakerSTTHandler

    fixed_stt = MagicMock(name="stt_fixed")

    _server, factory = asyncio.run(_call_start_wyoming_server(stt=fixed_stt, stt_provider=None))
    assert factory is not None

    r, w = _make_reader_writer()
    handler = factory(r, w)

    assert isinstance(handler, SpeakerSTTHandler)
    assert handler._stt is fixed_stt


# ---------------------------------------------------------------------------
# STT speaker_store_provider — resolved per _process_audio call, not just
# per connection (config-apply can rebind _state["speaker_store"] mid-call),
# mirroring the TTS speaker_store_provider pattern below.
# ---------------------------------------------------------------------------


def test_stt_speaker_store_provider_threaded_into_handler():
    """speaker_store_provider reaches the SpeakerSTTHandler constructed by
    the factory."""
    import asyncio

    provider = MagicMock(name="speaker_store_provider")

    _server, factory = asyncio.run(_call_start_wyoming_server(speaker_store_provider=provider))
    assert factory is not None

    r, w = _make_reader_writer()
    handler = factory(r, w)

    assert handler._speaker_store_provider is provider


def test_stt_no_speaker_store_provider_resolves_to_none():
    """``speaker_store_provider=None`` (no store configured) resolves to
    ``None`` — there is no static ``speaker_store=`` alternative on
    ``SpeakerSTTHandler``/``start_wyoming_server``; no production caller
    ever needed one (``paramem.server.app``'s only call site always
    passes a provider)."""
    import asyncio

    _server, factory = asyncio.run(_call_start_wyoming_server(speaker_store_provider=None))
    assert factory is not None

    r, w = _make_reader_writer()
    handler = factory(r, w)

    assert handler._speaker_store_provider is None


def test_stt_process_audio_resolves_speaker_store_via_provider_on_every_call():
    """``_process_audio``'s ``compute_embedding`` gate re-resolves the
    speaker store via the provider on EACH call — not just once at
    connection/handler construction — so a config-apply that rebinds
    ``_state["speaker_store"]`` mid-connection is picked up on the very
    next utterance."""
    import asyncio

    from paramem.server.wyoming_handler import SpeakerSTTHandler

    store_v1 = None
    store_v2 = MagicMock(name="speaker_store_v2")
    stores = [store_v1, store_v2]
    call_count = 0

    def provider():
        nonlocal call_count
        result = stores[min(call_count, len(stores) - 1)]
        call_count += 1
        return result

    stt = MagicMock(name="stt")
    handler = SpeakerSTTHandler(
        reader=MagicMock(),
        writer=MagicMock(),
        stt=stt,
        speaker_store_provider=provider,
    )
    handler._audio_buffer = bytearray(b"\x00\x00" * 800)

    captured_compute_embedding = []

    async def fake_process_utterance(*args, **kwargs):
        captured_compute_embedding.append(kwargs.get("compute_embedding"))
        result = MagicMock()
        result.text = "hello"
        result.language = "en"
        result.language_probability = 0.9
        result.embedding = None
        return result

    async def _fake_write(event, writer):
        return None

    with (
        patch("paramem.server.wyoming_handler.process_utterance", fake_process_utterance),
        patch("paramem.server.wyoming_handler.async_write_event", _fake_write),
    ):
        asyncio.run(handler._process_audio())
        asyncio.run(handler._process_audio())

    assert captured_compute_embedding == [False, True], (
        "compute_embedding must reflect the provider's CURRENT return value "
        "on each call, not a value cached at handler construction"
    )


def test_tts_manager_provider_called_per_handler_factory_invocation():
    """tts_manager_provider is called on every TTS handler-factory invocation."""
    import asyncio

    from paramem.server.wyoming_handler import TTSHandler

    mgr_v1 = MagicMock(name="tts_gpu")
    mgr_v2 = MagicMock(name="tts_cpu")
    instances = [mgr_v1, mgr_v2]
    call_count = 0

    def provider():
        nonlocal call_count
        result = instances[min(call_count, len(instances) - 1)]
        call_count += 1
        return result

    _server, factory = asyncio.run(_call_start_wyoming_tts_server(tts_manager_provider=provider))
    assert factory is not None

    r1, w1 = _make_reader_writer()
    r2, w2 = _make_reader_writer()

    handler1 = factory(r1, w1)
    handler2 = factory(r2, w2)

    assert isinstance(handler1, TTSHandler)
    assert isinstance(handler2, TTSHandler)
    assert call_count == 2
    assert handler1._tts is mgr_v1
    assert handler2._tts is mgr_v2


def test_back_compat_tts_manager_param_still_works():
    """When tts_manager_provider is None, the captured tts_manager param is used."""
    import asyncio

    from paramem.server.wyoming_handler import TTSHandler

    fixed_mgr = MagicMock(name="tts_fixed")

    _server, factory = asyncio.run(
        _call_start_wyoming_tts_server(tts_manager=fixed_mgr, tts_manager_provider=None)
    )
    assert factory is not None

    r, w = _make_reader_writer()
    handler = factory(r, w)

    assert isinstance(handler, TTSHandler)
    assert handler._tts is fixed_mgr


# ---------------------------------------------------------------------------
# speaker_store_provider — resolved per _synthesize_and_send call, not just
# per connection (config-apply can rebind _state["speaker_store"] mid-call).
# ---------------------------------------------------------------------------


def test_speaker_store_provider_threaded_into_tts_handler():
    """speaker_store_provider reaches the TTSHandler constructed by the factory."""
    import asyncio

    provider = MagicMock(name="speaker_store_provider")

    _server, factory = asyncio.run(_call_start_wyoming_tts_server(speaker_store_provider=provider))
    assert factory is not None

    r, w = _make_reader_writer()
    handler = factory(r, w)

    assert handler._speaker_store_provider is provider


def test_no_speaker_store_provider_resolves_to_none():
    """``speaker_store_provider=None`` (no store configured) resolves to
    ``None`` — there is no static ``speaker_store=`` alternative on
    ``TTSHandler``/``start_wyoming_tts_server``; no production caller
    ever needed one (``paramem.server.app``'s only call site always
    passes a provider)."""
    import asyncio

    _server, factory = asyncio.run(_call_start_wyoming_tts_server(speaker_store_provider=None))
    assert factory is not None

    r, w = _make_reader_writer()
    handler = factory(r, w)

    assert handler._speaker_store_provider is None


def test_synthesize_resolves_speaker_store_via_provider_on_every_call():
    """_synthesize_and_send re-resolves the speaker store via the provider on
    EACH call — not just once at connection/handler construction — so a
    config-apply that rebinds ``_state["speaker_store"]`` mid-connection is
    picked up on the very next synthesize."""
    import asyncio

    from paramem.server.wyoming_handler import TTSHandler

    store_v1 = MagicMock(name="speaker_store_v1")
    store_v2 = MagicMock(name="speaker_store_v2")
    stores = [store_v1, store_v2]
    call_count = 0

    def provider():
        nonlocal call_count
        result = stores[min(call_count, len(stores) - 1)]
        call_count += 1
        return result

    tts = MagicMock(name="tts_manager")
    tts.needs_gpu.return_value = False
    tts.synthesize.return_value = (b"\x00\x00" * 10, 24000)

    handler = TTSHandler(
        reader=MagicMock(),
        writer=MagicMock(),
        tts_manager=tts,
        speaker_store_provider=provider,
    )

    resolved_calls = []

    def fake_resolve(text, store, **kwargs):
        resolved_calls.append(store)
        return text

    async def _fake_write(event, writer):
        return None

    with (
        patch("paramem.server.wyoming_handler.resolve_speaker_tokens", fake_resolve),
        patch("paramem.server.wyoming_handler.async_write_event", _fake_write),
    ):
        asyncio.run(handler._synthesize_and_send("hello", None))
        asyncio.run(handler._synthesize_and_send("world", None))

    assert resolved_calls == [store_v1, store_v2]


def test_synthesize_streaming_join_path_resolves_via_provider():
    """The streaming-join path (SynthesizeStop rendering the accumulated
    chunk text) also goes through the same defensive resolve, threaded
    through the provider."""
    import asyncio

    from wyoming.tts import SynthesizeChunk, SynthesizeStart, SynthesizeStop

    from paramem.server.wyoming_handler import TTSHandler

    store = MagicMock(name="speaker_store")
    provider = MagicMock(return_value=store)

    tts = MagicMock(name="tts_manager")
    tts.needs_gpu.return_value = False
    tts.synthesize.return_value = (b"\x00\x00" * 10, 24000)

    handler = TTSHandler(
        reader=MagicMock(),
        writer=MagicMock(),
        tts_manager=tts,
        speaker_store_provider=provider,
    )

    resolved_calls = []

    def fake_resolve(text, resolved_store, **kwargs):
        resolved_calls.append((text, resolved_store))
        return text

    async def _drive():
        async def _fake_write(event, writer):
            return None

        with patch("paramem.server.wyoming_handler.async_write_event", _fake_write):
            await handler.handle_event(SynthesizeStart(voice=None).event())
            await handler.handle_event(SynthesizeChunk(text="hi there").event())
            await handler.handle_event(SynthesizeStop().event())

    with patch("paramem.server.wyoming_handler.resolve_speaker_tokens", fake_resolve):
        asyncio.run(_drive())

    assert resolved_calls == [("hi there", store)]
    provider.assert_called_once()


def test_synthesize_second_pass_leaves_unresolvable_token_verbatim():
    """Real (UNMOCKED) ``resolve_speaker_tokens`` path — the Wyoming TTS
    handler's second application must pass
    ``unresolvable_fallback="verbatim"``: an unresolvable ``speaker{N}``
    token in the synthesize text survives verbatim into what actually
    reaches ``TTSManager.synthesize`` (never collapsed to
    THIRD_PARTY_DESCRIPTOR, which would narrate a contract violation as
    "another speaker" IN SPEECH), while a token the store CAN resolve still
    resolves normally."""
    import asyncio

    from paramem.server.wyoming_handler import TTSHandler

    class _StubStore:
        def resolve_speaker_name(self, token):
            return "Alice" if token == "speaker0" else None

    tts = MagicMock(name="tts_manager")
    tts.needs_gpu.return_value = False
    tts.synthesize.return_value = (b"\x00\x00" * 10, 24000)

    handler = TTSHandler(
        reader=MagicMock(),
        writer=MagicMock(),
        tts_manager=tts,
        speaker_store_provider=lambda: _StubStore(),
    )

    async def _fake_write(event, writer):
        return None

    with patch("paramem.server.wyoming_handler.async_write_event", _fake_write):
        asyncio.run(handler._synthesize_and_send("speaker0 and speaker9 are meeting.", None))

    synthesized_text = tts.synthesize.call_args[0][0]
    assert synthesized_text == "Alice and speaker9 are meeting."
