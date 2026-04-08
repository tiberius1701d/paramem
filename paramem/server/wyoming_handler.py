"""Wyoming protocol handler for STT with speaker embedding.

Receives audio from HA voice satellites via the Wyoming protocol,
transcribes using local Whisper, computes speaker embeddings,
and returns the transcript.
"""

import asyncio
import logging

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_write_event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

logger = logging.getLogger(__name__)

# Shared GPU lock — prevents concurrent Whisper and LLM inference.
# Imported and used by app.py for LLM inference as well.
gpu_inference_lock = asyncio.Lock()


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
    ):
        super().__init__(reader, writer)
        self._stt = stt
        self._speaker_store = speaker_store
        self._chat_callback = chat_callback
        self._embedding_callback = embedding_callback
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

        # Transcribe with GPU lock to prevent concurrent GPU access with LLM
        loop = asyncio.get_running_loop()
        async with gpu_inference_lock:
            text = await loop.run_in_executor(
                None, self._stt.transcribe, audio_bytes, self._sample_rate
            )

        logger.info("Transcript: '%s'", text[:100] if text else "(empty)")

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
        """Compute speaker embedding from audio on CPU."""
        try:
            from paramem.server.speaker_embedding import compute_speaker_embedding

            embedding = compute_speaker_embedding(audio_bytes, self._sample_rate)
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
            else ["en", "de", "fr", "es", "it", "pt", "nl", "ja", "zh"]
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
) -> AsyncServer:
    """Start the Wyoming STT server (non-blocking).

    Returns the server instance. Call server.stop() on shutdown.
    """

    def handler_factory(reader, writer):
        return SpeakerSTTHandler(
            reader, writer, stt, speaker_store, chat_callback, embedding_callback
        )

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")

    logger.info("Wyoming STT server starting on %s:%d", host, port)
    await server.start(handler_factory)

    return server
