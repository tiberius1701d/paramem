"""Tests for Wyoming handler factory provider-callable indirection.

Verifies that ``start_wyoming_server`` and ``start_wyoming_tts_server``
call the provider callable on every handler factory invocation so that
voice-profile hot-swaps (gpu/cpu) take effect without restarting the
socket listeners.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_reader_writer():
    """Minimal asyncio stream pair stubs."""
    return MagicMock(name="reader"), MagicMock(name="writer")


async def _call_start_wyoming_server(stt=None, stt_provider=None):
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
        )

    return server, captured_factory.get("fn")


async def _call_start_wyoming_tts_server(tts_manager=None, tts_manager_provider=None):
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
