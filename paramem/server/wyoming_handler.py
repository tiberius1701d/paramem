"""Wyoming protocol handlers for STT and TTS.

STT: Receives audio from HA voice satellites, transcribes via Whisper,
computes speaker embeddings, detects language.

TTS: Receives text from HA, synthesizes speech via Piper or MMS-TTS
in the detected language, returns audio.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_write_event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from paramem.server.config import ISO_LANGUAGE_NAMES
from paramem.server.speaker import resolve_speaker_tokens
from paramem.server.voice_pipeline import process_utterance

if TYPE_CHECKING:
    from paramem.server.tts import TTSManager

logger = logging.getLogger(__name__)

# GPU lock — thread-safe, protects against concurrent CUDA access
from paramem.server.gpu_lock import gpu_lock as _gpu_lock  # noqa: E402


class SpeakerSTTHandler(AsyncEventHandler):
    """Handles a single Wyoming STT connection.

    Accumulates audio chunks, transcribes via Whisper, optionally
    computes a speaker embedding, and returns the transcript.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        stt,
        speaker_store_provider: Callable[[], object] | None = None,
        chat_callback=None,
        embedding_callback=None,
        language_callback=None,
        min_embedding_duration_seconds: float = 1.0,
    ):
        super().__init__(reader, writer)
        self._stt = stt
        # Re-resolved on every _process_audio call (not just once at
        # connection-open) — mirrors TTSHandler's speaker_store_provider
        # pattern: a live Wyoming connection can outlive a config-apply
        # that rebinds ``_state["speaker_store"]`` from ``None`` to a real
        # store (or back), and an eagerly-captured reference would keep
        # gating embedding computation on the stale value indefinitely.
        # ``None`` is a valid provider (no speaker store configured).
        self._speaker_store_provider = speaker_store_provider
        self._chat_callback = chat_callback
        self._embedding_callback = embedding_callback
        self._language_callback = language_callback
        self._min_embedding_duration_seconds = min_embedding_duration_seconds
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        self._sample_width = 2
        self._channels = 1

    async def handle_event(self, event: Event) -> bool:
        """Process a Wyoming protocol event. Returns True to continue."""
        if Describe.is_type(event.type):
            await self._send_info()
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._audio_buffer = bytearray()
            self._sample_rate = start.rate
            self._sample_width = start.width
            self._channels = start.channels
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            await self._process_audio()
            return False  # Connection complete

        return True

    async def _process_audio(self) -> None:
        """Transcribe accumulated audio and dispatch Wyoming + ParaMem callbacks.

        The GPU/CPU core (empty-buffer guard, STT-not-loaded guard, transcription
        under the GPU lock, and optional speaker-embedding computation) is
        delegated to :func:`~paramem.server.voice_pipeline.process_utterance`.
        This method handles only the Wyoming-specific parts: writing the
        ``Transcript`` event to the TCP stream and invoking the server-state
        callbacks (language detection, embedding store, chat forwarding).
        The callback / write order is preserved:
          1. Language callback (propagate detected language to server state).
          2. Embedding callback (store latest embedding in server state).
          3. ``Transcript`` write (send result back via Wyoming protocol).
          4. Chat callback (forward transcript + embedding to ParaMem /chat).

        The speaker store is re-resolved via ``self._speaker_store_provider``
        on THIS call (not cached at connection-open) so a config-apply that
        rebinds ``_state["speaker_store"]`` mid-connection is picked up by
        the very next utterance's ``compute_embedding`` gate.
        """
        audio_bytes = bytes(self._audio_buffer)
        speaker_store = (
            self._speaker_store_provider() if self._speaker_store_provider is not None else None
        )

        utterance = await process_utterance(
            audio_bytes,
            self._sample_rate,
            self._sample_width,
            self._channels,
            self._stt,
            compute_embedding=speaker_store is not None,
            min_embedding_duration_seconds=self._min_embedding_duration_seconds,
        )

        text = utterance.text

        # 1. Propagate detected language to server state
        if self._language_callback and text:
            self._language_callback(utterance.language, utterance.language_probability)

        # 2. Store latest embedding in server state
        if utterance.embedding is not None and self._embedding_callback:
            self._embedding_callback(utterance.embedding)

        # 3. Send transcript back via Wyoming protocol
        await async_write_event(Transcript(text=text or "").event(), self.writer)

        # 4. Forward transcript + embedding to ParaMem
        if self._chat_callback and text:
            await self._chat_callback(text, utterance.embedding)

    async def _send_info(self) -> None:
        """Respond to Describe event with service info."""
        languages = (
            [self._stt.language]
            if self._stt.language != "auto"
            else list(ISO_LANGUAGE_NAMES.keys())
        )
        info = Info(
            asr=[
                AsrProgram(
                    name="paramem-whisper",
                    description="ParaMem local Whisper STT",
                    attribution=Attribution(
                        name="ParaMem",
                        url="https://github.com/tiberius1701d/paramem",
                    ),
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name=self._stt.model_name,
                            description=f"Whisper {self._stt.model_name}",
                            attribution=Attribution(
                                name="OpenAI",
                                url="https://github.com/openai/whisper",
                            ),
                            installed=True,
                            version="1.0.0",
                            languages=languages,
                        )
                    ],
                )
            ],
        )
        await async_write_event(info.event(), self.writer)


async def start_wyoming_server(
    host: str,
    port: int,
    stt=None,
    speaker_store_provider: Callable[[], object] | None = None,
    chat_callback=None,
    embedding_callback=None,
    language_callback=None,
    min_embedding_duration_seconds: float = 1.0,
    stt_provider: Callable[[], object] | None = None,
) -> AsyncServer:
    """Start the Wyoming STT server (non-blocking).

    Args:
        host: TCP host to bind.
        port: TCP port to listen on.
        stt: Loaded STT model instance. Used when ``stt_provider`` is None.
        speaker_store_provider: Callable returning the active
            :class:`~paramem.server.speaker.SpeakerStore` (or ``None`` when
            no store is configured), used to gate speaker-embedding
            computation. Threaded straight through to
            :class:`SpeakerSTTHandler`, which re-resolves it on every
            utterance (not just once per connection) — the lifespan boot
            path's only production caller, mirroring
            :func:`start_wyoming_tts_server`'s ``speaker_store_provider``,
            so a live config-apply that rebinds ``_state["speaker_store"]``
            takes effect without restarting the socket listener or
            dropping an in-flight connection. There is no static
            ``speaker_store=`` alternative — no production caller ever
            needed one.
        chat_callback: Async callable forwarding (text, embedding) to /chat.
        embedding_callback: Callable storing the latest embedding in server state.
        language_callback: Callable storing the detected language in server state.
        min_embedding_duration_seconds: Minimum audio duration to compute an
            embedding; passed through to compute_speaker_embedding().
        stt_provider: Optional callable returning the active STT instance. When
            provided, the handler factory calls this on every connection so
            profile hot-swaps (gpu/cpu) take effect without restarting the
            socket listener. Supersedes ``stt`` when not None.

    Returns the server instance. Call server.stop() on shutdown.
    """

    def handler_factory(reader, writer):
        active_stt = stt_provider() if stt_provider is not None else stt
        return SpeakerSTTHandler(
            reader,
            writer,
            active_stt,
            speaker_store_provider,
            chat_callback,
            embedding_callback,
            language_callback,
            min_embedding_duration_seconds=min_embedding_duration_seconds,
        )

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")

    logger.info("Wyoming STT server starting on %s:%d", host, port)
    await server.start(handler_factory)

    return server


# ---------------------------------------------------------------------------
# TTS handler
# ---------------------------------------------------------------------------


def _resolve_synth_language(hint: str | None, detected: str | None, source: str) -> str | None:
    """Pick the TTS synth language from the caller's voice hint and ParaMem's
    detected language, per ``tts.language_source``. ``"hint"`` lets the caller's
    hint win; ``"auto"``/``"detection"`` let the detected language win. Both fall
    back to the other source (then to the TTSManager default downstream)."""
    if source == "hint":
        return hint or detected
    return detected or hint


class TTSHandler(AsyncEventHandler):
    """Handles a single Wyoming TTS connection.

    Receives a Synthesize event with text, synthesizes audio in the
    detected language, and returns audio chunks.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        tts_manager: TTSManager,
        language_resolver=None,
        audio_chunk_bytes: int = 4096,
        language_source: str = "auto",
        speaker_store_provider: Callable[[], object] | None = None,
    ):
        super().__init__(reader, writer)
        self._tts = tts_manager
        self._language_resolver = language_resolver
        self._audio_chunk_bytes = audio_chunk_bytes
        self._language_source = language_source
        # Re-resolved on every _synthesize_and_send call (not just once at
        # connection-open) — mirrors the tts_manager_provider pattern but at
        # call granularity: a live Wyoming connection can outlive a
        # config-apply that rebinds ``_state["speaker_store"]`` to a fresh
        # SpeakerStore instance, and an eagerly-captured reference would keep
        # resolving against the stale (possibly-None) one indefinitely.
        # ``None`` is a valid provider (no speaker store configured) —
        # resolve_speaker_tokens already accepts a ``None`` store.
        self._speaker_store_provider = speaker_store_provider
        # Accumulated state for streaming synthesis (SynthesizeStart/Chunk/Stop).
        # ``_streaming`` is True between a SynthesizeStart and the terminal event
        # that completes the stream — so the response can be finalized with a
        # SynthesizeStopped (which HA's streaming path requires) instead of a
        # bare connection close.
        self._streaming = False
        self._stream_voice = None
        self._stream_text: list[str] = []
        # True once audio for the current stream has been sent (via the terminal
        # Synthesize), so the trailing SynthesizeStop only emits SynthesizeStopped
        # and does not re-synthesize the same utterance.
        self._audio_sent = False

    async def handle_event(self, event: Event) -> bool:
        """Process a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self._send_info()
            return True

        # Synthesize carries the complete text. HA's streaming voice pipeline
        # sends it mid-stream (SynthesizeStart -> chunk(s) -> Synthesize ->
        # SynthesizeStop) as the consolidated full utterance; non-streaming
        # callers (tts.speak, tts_get_url) send it alone. Inside a stream we send
        # the audio now but DEFER SynthesizeStopped to SynthesizeStop — HA's
        # streaming state machine expects SynthesizeStopped only AFTER it sends
        # the stop (matches wyoming-piper); emitting it early makes HA drop the
        # audio (satellite/Sonos blinks, stays silent). A non-streaming caller
        # treats connection-close as completion, so keep that path closing.
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            voice = synthesize.voice if synthesize.voice is not None else self._stream_voice
            await self._synthesize_and_send(synthesize.text, voice)
            if self._streaming:
                self._audio_sent = True
                return True
            return False  # one-shot: connection-close signals completion

        # Streaming synthesis (HA voice pipeline): SynthesizeStart -> chunk(s) ->
        # [Synthesize] -> SynthesizeStop. HA streams the response text token-by-
        # token; our engines are not incremental, so the full text arrives either
        # in the consolidated Synthesize (above) or accumulated across chunks
        # (rendered on stop below). Advertising this path
        # (supports_synthesize_streaming) is what makes HA deliver our audio to
        # the satellite/Sonos the same way it does for wyoming-piper.
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._streaming = True
            self._stream_voice = start.voice
            self._stream_text = []
            self._audio_sent = False
            logger.info("TTS stream: START (voice=%s)", start.voice.name if start.voice else None)
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            self._stream_text.append(chunk.text)
            return True

        # SynthesizeStop ends the stream. If the consolidated Synthesize already
        # sent the audio, only emit SynthesizeStopped; otherwise render the
        # accumulated chunk text first (HA builds that stream without a
        # consolidated Synthesize). The ``_audio_sent`` guard prevents
        # double-synthesizing the same utterance.
        if SynthesizeStop.is_type(event.type):
            if not self._streaming:
                logger.info("TTS stream: STOP received outside a stream (no-op)")
                return True
            if not self._audio_sent:
                full = "".join(self._stream_text)
                logger.info(
                    "TTS stream: STOP (%d chunks, %d chars)", len(self._stream_text), len(full)
                )
                await self._synthesize_and_send(full, self._stream_voice)
            await async_write_event(SynthesizeStopped().event(), self.writer)
            self._reset_stream()
            logger.info("TTS stream: STOPPED sent")
            return True

        logger.info("TTS handler: unhandled event type=%s", event.type)
        return True

    def _reset_stream(self) -> None:
        """Clear per-stream accumulation after a stream is finalized."""
        self._streaming = False
        self._stream_text = []
        self._stream_voice = None
        self._audio_sent = False

    async def _synthesize_and_send(self, text: str, voice) -> None:
        """Synthesize ``text`` and stream the audio back via the Wyoming
        protocol. Shared by the one-shot Synthesize path and the streaming
        SynthesizeStart/Chunk/Stop path; ``voice`` is the caller's
        SynthesizeVoice (or None).

        Defensive SECOND application of :func:`~paramem.server.speaker.
        resolve_speaker_tokens` — text reaching this method should already
        have its ``speaker{N}`` tokens resolved to display names by the
        reply-boundary resolver at the app layer, but a caller could in
        principle route unresolved text straight to TTS (a satellite/HA
        integration path that bypasses ``/chat``/``/voice`` entirely, or a
        future contract violation).  Unlike the app-layer resolver, this
        call uses ``unresolvable_fallback="verbatim"``, NOT the default
        ``"descriptor"`` collapse: anything still carrying a raw
        ``speaker{N}`` token by the time it reaches TTS synthesis is either
        (a) the caller's own self-token (deliberately left verbatim by the
        app-layer resolver, harmless to repeat here) or (b) a genuine
        contract violation — unresolved text that skipped the app-layer
        resolver.  Collapsing case (b) to :data:`~paramem.server.speaker.
        THIRD_PARTY_DESCRIPTOR` would silently narrate a real self-reference
        as "another speaker" IN SPEECH — worse than the honest fail-safe of
        leaving the raw token audible, which is what ``"verbatim"``
        produces.  ``current_speaker_id`` is still omitted here — the
        Wyoming protocol carries no notion of "the current speaker" at this
        layer, and the app-layer resolution that already ran is the one
        call site that knows it — but it is no longer needed for the
        self-token case: ``"verbatim"`` already leaves ANY unresolvable
        token untouched, self-token or not.

        The speaker store is re-resolved via ``self._speaker_store_provider``
        on every call (``None`` provider resolves to no store, matching
        ``resolve_speaker_tokens``'s own ``None``-store contract) so a
        config-apply that rebinds ``_state["speaker_store"]`` mid-connection
        is picked up on the very next synthesize, including the
        streaming-join path (``SynthesizeStop`` rendering the accumulated
        chunk text).
        """
        if not text:
            logger.warning("Empty TTS request")
            return

        speaker_store = (
            self._speaker_store_provider() if self._speaker_store_provider is not None else None
        )
        text = resolve_speaker_tokens(text, speaker_store, unresolvable_fallback="verbatim")

        # Resolve language per tts.language_source. Detection-first ("auto"/
        # "detection") prefers ParaMem's detected language over the caller's
        # voice.language hint; "hint" reverses it. Both fall back to the other
        # source, then to the TTSManager default_language.
        hint = voice.language if (voice and voice.language) else None
        detected = self._language_resolver() if self._language_resolver else None
        language = _resolve_synth_language(hint, detected, self._language_source)

        logger.info(
            "TTS request: '%s' (lang=%s)",
            text[:80],
            language or "default",
        )

        loop = asyncio.get_running_loop()

        # Acquire GPU lock only if THIS language's engine is on GPU
        try:
            if self._tts.needs_gpu(language):
                async with _gpu_lock():
                    pcm_data, sample_rate = await loop.run_in_executor(
                        None, self._tts.synthesize, text, language
                    )
            else:
                pcm_data, sample_rate = await loop.run_in_executor(
                    None, self._tts.synthesize, text, language
                )
        except Exception:
            logger.exception("TTS synthesis failed for lang=%s", language)
            return

        # Send audio back via Wyoming protocol
        await async_write_event(
            AudioStart(rate=sample_rate, width=2, channels=1).event(),
            self.writer,
        )

        # Send in chunks (4096 bytes ~ 128ms at 16kHz)
        chunk_size = self._audio_chunk_bytes
        for i in range(0, len(pcm_data), chunk_size):
            await async_write_event(
                AudioChunk(
                    audio=pcm_data[i : i + chunk_size],
                    rate=sample_rate,
                    width=2,
                    channels=1,
                ).event(),
                self.writer,
            )

        await async_write_event(AudioStop().event(), self.writer)
        logger.debug("TTS complete: %d bytes audio", len(pcm_data))

    async def _send_info(self) -> None:
        """Respond to Describe event with TTS service info."""
        voices = [
            TtsVoice(
                name=f"paramem-{lang}",
                description=f"ParaMem TTS ({lang})",
                attribution=Attribution(
                    name="ParaMem",
                    url="https://github.com/tiberius1701d/paramem",
                ),
                installed=True,
                version="1.0.0",
                languages=[lang],
            )
            for lang in self._tts.available_languages
        ]
        info = Info(
            tts=[
                TtsProgram(
                    name="paramem-tts",
                    description="ParaMem multilingual TTS",
                    attribution=Attribution(
                        name="ParaMem",
                        url="https://github.com/tiberius1701d/paramem",
                    ),
                    installed=True,
                    version="1.0.0",
                    voices=voices,
                    supports_synthesize_streaming=True,
                )
            ],
        )
        await async_write_event(info.event(), self.writer)


async def start_wyoming_tts_server(
    host: str,
    port: int,
    tts_manager: TTSManager | None = None,
    language_resolver=None,
    audio_chunk_bytes: int = 4096,
    tts_manager_provider: Callable[[], TTSManager] | None = None,
    language_source: str = "auto",
    speaker_store_provider: Callable[[], object] | None = None,
) -> AsyncServer:
    """Start the Wyoming TTS server (non-blocking).

    Args:
        tts_manager: Loaded TTSManager with voice engines. Used when
            ``tts_manager_provider`` is None.
        language_resolver: Callable returning the detected language code.
        audio_chunk_bytes: Bytes per Wyoming audio chunk sent to satellite.
        tts_manager_provider: Optional callable returning the active TTSManager.
            When provided, the handler factory calls this on every connection so
            profile hot-swaps (gpu/cpu) take effect without restarting the
            socket listener. Supersedes ``tts_manager`` when not None.
        language_source: ``tts.language_source`` — "auto"/"detection" prefer the
            detected language over the caller's voice hint; "hint" reverses it.
        speaker_store_provider: Callable returning the active
            :class:`~paramem.server.speaker.SpeakerStore` (or ``None`` when
            no store is configured), used for the defensive reply-boundary
            resolve in :meth:`TTSHandler._synthesize_and_send`.  Threaded
            straight through to :class:`TTSHandler`, which re-resolves it on
            every ``_synthesize_and_send`` call (not just once per
            connection) — the lifespan boot path's only production caller
            (mirrors ``tts_manager_provider``) so a live config-apply that
            rebinds ``_state["speaker_store"]`` takes effect without
            restarting the socket listener or dropping an in-flight
            connection.  There is no static ``speaker_store=`` alternative —
            no production caller ever needed one.
    """

    def handler_factory(reader, writer):
        active_tts = tts_manager_provider() if tts_manager_provider is not None else tts_manager
        return TTSHandler(
            reader,
            writer,
            active_tts,
            language_resolver,
            audio_chunk_bytes,
            language_source,
            speaker_store_provider,
        )

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")

    logger.info("Wyoming TTS server starting on %s:%d", host, port)
    await server.start(handler_factory)

    return server
