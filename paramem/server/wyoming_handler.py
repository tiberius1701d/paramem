"""Wyoming protocol handlers for STT and TTS.

STT: Receives audio from HA voice satellites, transcribes via Whisper,
computes speaker embeddings, detects language.

TTS: Receives text from HA, synthesizes speech via Piper or MMS-TTS
in the detected language, returns audio.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_write_event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

from paramem.server.config import ISO_LANGUAGE_NAMES

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
        """Transcribe accumulated audio and return result."""
        if not self._audio_buffer:
            logger.warning("Empty audio buffer — nothing to transcribe")
            await async_write_event(Transcript(text="").event(), self.writer)
            return

        # Check if STT is still loaded (may have been released for GPU)
        if self._stt is None or not self._stt.is_loaded:
            logger.warning("STT model not loaded — returning empty transcript")
            await async_write_event(Transcript(text="").event(), self.writer)
            return

        audio_bytes = bytes(self._audio_buffer)
        duration = len(audio_bytes) / (self._sample_rate * self._sample_width * self._channels)
        logger.info(
            "Processing %d bytes of audio (%.1fs at %dHz)",
            len(audio_bytes),
            duration,
            self._sample_rate,
        )

        # Validate audio format
        if self._sample_width != 2:
            logger.warning(
                "Unexpected sample width %d (expected 2 for int16)",
                self._sample_width,
            )
        if self._channels != 1:
            logger.warning(
                "Unexpected channel count %d (expected 1 for mono)",
                self._channels,
            )

        # Transcribe with GPU lock to prevent concurrent GPU access with LLM/training
        loop = asyncio.get_running_loop()
        async with _gpu_lock():
            result = await loop.run_in_executor(
                None, self._stt.transcribe, audio_bytes, self._sample_rate
            )

        text = result.text
        logger.info(
            "Transcript: '%s' (lang=%s, prob=%.2f)",
            text[:100] if text else "(empty)",
            result.language,
            result.language_probability,
        )

        # Propagate detected language to server state
        if self._language_callback and text:
            self._language_callback(result.language, result.language_probability)

        # Compute speaker embedding on CPU (no GPU lock needed)
        embedding = None
        if self._speaker_store is not None:
            embedding = await loop.run_in_executor(None, self._compute_embedding, audio_bytes)
            if embedding and self._embedding_callback:
                self._embedding_callback(embedding)

        # Send transcript back via Wyoming protocol
        await async_write_event(Transcript(text=text or "").event(), self.writer)

        # If we have a chat callback, forward transcript + embedding to ParaMem
        if self._chat_callback and text:
            await self._chat_callback(text, embedding)

    def _compute_embedding(self, audio_bytes: bytes) -> list[float] | None:
        """Compute speaker embedding from audio on CPU.

        Duration filtering is delegated to compute_speaker_embedding() via
        min_duration_seconds so the gate is based on audio length, not word count.
        """
        try:
            from paramem.server.speaker_embedding import compute_speaker_embedding

            embedding = compute_speaker_embedding(
                audio_bytes,
                self._sample_rate,
                min_duration_seconds=self._min_embedding_duration_seconds,
            )
            return embedding if embedding else None
        except ImportError:
            logger.debug("Speaker embedding not available — install paramem[speaker]")
            return None
        except Exception:
            logger.warning("Speaker embedding computation failed", exc_info=True)
            return None

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
    stt,
    speaker_store=None,
    chat_callback=None,
    embedding_callback=None,
    language_callback=None,
    min_embedding_duration_seconds: float = 1.0,
) -> AsyncServer:
    """Start the Wyoming STT server (non-blocking).

    Args:
        host: TCP host to bind.
        port: TCP port to listen on.
        stt: Loaded STT model instance.
        speaker_store: Optional SpeakerStore for embedding enrichment.
        chat_callback: Async callable forwarding (text, embedding) to /chat.
        embedding_callback: Callable storing the latest embedding in server state.
        language_callback: Callable storing the detected language in server state.
        min_embedding_duration_seconds: Minimum audio duration to compute an
            embedding; passed through to compute_speaker_embedding().

    Returns the server instance. Call server.stop() on shutdown.
    """

    def handler_factory(reader, writer):
        return SpeakerSTTHandler(
            reader,
            writer,
            stt,
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
    ):
        super().__init__(reader, writer)
        self._tts = tts_manager
        self._language_resolver = language_resolver
        self._audio_chunk_bytes = audio_chunk_bytes

    async def handle_event(self, event: Event) -> bool:
        """Process a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self._send_info()
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            await self._synthesize(synthesize)
            return False  # Connection complete

        return True

    async def _synthesize(self, request: Synthesize) -> None:
        """Synthesize text and send audio back."""
        text = request.text
        if not text:
            logger.warning("Empty TTS request")
            return

        # Resolve language: voice hint → resolver callback → default
        language = None
        if request.voice and request.voice.language:
            language = request.voice.language
        elif self._language_resolver:
            language = self._language_resolver()

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
                )
            ],
        )
        await async_write_event(info.event(), self.writer)


async def start_wyoming_tts_server(
    host: str,
    port: int,
    tts_manager: TTSManager,
    language_resolver=None,
    audio_chunk_bytes: int = 4096,
) -> AsyncServer:
    """Start the Wyoming TTS server (non-blocking).

    Args:
        tts_manager: Loaded TTSManager with voice engines.
        language_resolver: Callable returning the detected language code.
        audio_chunk_bytes: Bytes per Wyoming audio chunk sent to satellite.
    """

    def handler_factory(reader, writer):
        return TTSHandler(reader, writer, tts_manager, language_resolver, audio_chunk_bytes)

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")

    logger.info("Wyoming TTS server starting on %s:%d", host, port)
    await server.start(handler_factory)

    return server
