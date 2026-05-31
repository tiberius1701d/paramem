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
        speaker_store=None,
        chat_callback=None,
        embedding_callback=None,
        language_callback=None,
        min_embedding_duration_seconds: float = 1.0,
    ):
        super().__init__(reader, writer)
        self._stt = stt
        self._speaker_store = speaker_store
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
        """
        audio_bytes = bytes(self._audio_buffer)

        utterance = await process_utterance(
            audio_bytes,
            self._sample_rate,
            self._sample_width,
            self._channels,
            self._stt,
            compute_embedding=self._speaker_store is not None,
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
    speaker_store=None,
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
        speaker_store: Optional SpeakerStore for embedding enrichment.
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
            speaker_store,
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
    ):
        super().__init__(reader, writer)
        self._tts = tts_manager
        self._language_resolver = language_resolver
        self._audio_chunk_bytes = audio_chunk_bytes
        self._language_source = language_source
        # Accumulated state for streaming synthesis (SynthesizeStart/Chunk/Stop).
        self._stream_voice = None
        self._stream_text: list[str] = []

    async def handle_event(self, event: Event) -> bool:
        """Process a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self._send_info()
            return True

        # One-shot synthesis (tts.speak, tts_get_url, non-streaming callers).
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            await self._synthesize_and_send(synthesize.text, synthesize.voice)
            return False  # Connection complete

        # Streaming synthesis (HA voice pipeline): SynthesizeStart -> chunk(s) ->
        # SynthesizeStop. HA streams the response text token-by-token; we
        # accumulate it and render the full utterance on stop (our engines are
        # not incremental), then signal SynthesizeStopped. Advertising this path
        # (supports_synthesize_streaming) is what makes HA deliver our audio to
        # the satellite/Sonos the same way it does for wyoming-piper.
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._stream_voice = start.voice
            self._stream_text = []
            logger.info("TTS stream: START (voice=%s)", start.voice.name if start.voice else None)
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            self._stream_text.append(chunk.text)
            return True

        if SynthesizeStop.is_type(event.type):
            full = "".join(self._stream_text)
            logger.info("TTS stream: STOP (%d chunks, %d chars)", len(self._stream_text), len(full))
            await self._synthesize_and_send(full, self._stream_voice)
            await async_write_event(SynthesizeStopped().event(), self.writer)
            self._stream_text = []
            self._stream_voice = None
            logger.info("TTS stream: STOPPED sent")
            return True

        logger.info("TTS handler: unhandled event type=%s", event.type)
        return True

    async def _synthesize_and_send(self, text: str, voice) -> None:
        """Synthesize ``text`` and stream the audio back via the Wyoming
        protocol. Shared by the one-shot Synthesize path and the streaming
        SynthesizeStart/Chunk/Stop path; ``voice`` is the caller's
        SynthesizeVoice (or None)."""
        if not text:
            logger.warning("Empty TTS request")
            return

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
    """

    def handler_factory(reader, writer):
        active_tts = tts_manager_provider() if tts_manager_provider is not None else tts_manager
        return TTSHandler(
            reader, writer, active_tts, language_resolver, audio_chunk_bytes, language_source
        )

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")

    logger.info("Wyoming TTS server starting on %s:%d", host, port)
    await server.start(handler_factory)

    return server
