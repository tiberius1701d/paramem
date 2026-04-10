"""ParaMem server — REST API wrapping the parametric memory pipeline.

Usage:
    python -m paramem.server.app --config configs/server.yaml

GPU lifecycle (service-level):
    Stop service to free GPU, restart to reclaim.
    --defer-model: start without GPU model, auto-reclaim when GPU is free.
    --cloud-only: permanent cloud-only mode, no auto-reclaim.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from paramem.models.loader import load_adapter, load_base_model, unload_model
from paramem.server.cloud import get_cloud_agent
from paramem.server.config import TTSConfig, load_server_config
from paramem.server.consolidation import run_consolidation
from paramem.server.ha_graph import HAEntityGraph
from paramem.server.inference import ChatResult, _escalate_to_sota, handle_chat
from paramem.server.router import QueryRouter
from paramem.server.session_buffer import SessionBuffer
from paramem.server.tools.ha_client import HAClient
from paramem.server.tools.registry import ToolRegistry
from paramem.utils.notify import SERVER_CLOUD_ONLY, notify_server

logger = logging.getLogger(__name__)

# Resolve nvidia-smi at import time. On WSL2 it lives in /usr/lib/wsl/lib
# which systemd may not have on PATH; on native Linux it's typically in
# /usr/bin.  Falls back to bare name (will fail with FileNotFoundError,
# caught by the caller).
_NVIDIA_SMI = (
    shutil.which("nvidia-smi")
    or shutil.which("nvidia-smi", path="/usr/lib/wsl/lib:/usr/bin:/usr/local/bin")
    or "nvidia-smi"
)

# Global state — single model, single adapter, single server
_state = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "session_buffer": None,
    "router": None,
    "cloud_agent": None,
    "ha_client": None,
    "tool_registry": None,
    "consolidation_loop": None,
    "consolidating": False,
    "last_consolidation": None,
    "scheduler_task": None,
    "training_scheduler_task": None,
    "background_trainer": None,
    "reclaim_task": None,
    "mode": "local",  # "local" or "cloud-only"
    "cloud_only_reason": None,  # "explicit", "training", "gpu_conflict", or None
    "speaker_store": None,
    "stt": None,
    "wyoming_server": None,
    "tts_manager": None,
    "wyoming_tts_server": None,
    "latest_embedding": None,
    "latest_language_detection": None,  # {language: str, probability: float}
    "last_chat_time": None,
    "enrollment_task": None,
    "pending_enrollments": set(),
    # Unknown speaker groups: temp_id → {embeddings, conversations, first_seen}
    # Thread safety: both the chat handler and enrollment loop run on the asyncio
    # event loop (cooperative scheduling). Mutations are synchronous within each
    # handler, so no interleaving within a single dict operation. Safe without locks.
    "unknown_speakers": {},
    # Global cooldown: first prompt waits for the full interval after startup
    "last_enrollment_prompt": datetime.now(timezone.utc),
}


# --- Request/Response schemas ---


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"
    speaker: str | None = None
    speaker_embedding: list[float] | None = None  # Voice embedding from STT
    history: list[dict] | None = None
    route: str | None = None  # Force routing: "ha", "sota", or None (auto)


class ChatResponse(BaseModel):
    text: str
    escalated: bool = False
    speaker: str | None = None
    follow_up: str | None = None  # Server-initiated follow-up (e.g. introduction)


class StatusResponse(BaseModel):
    model: str
    mode: str  # "local" or "cloud-only"
    cloud_only_reason: str | None  # "explicit", "training", "gpu_conflict", or None
    adapter_loaded: bool
    keys_count: int
    pending_sessions: int
    consolidating: bool
    last_consolidation: str | None
    speaker_profiles: int = 0
    stt_loaded: bool = False
    stt_model: str | None = None


class ConsolidateResponse(BaseModel):
    status: str


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    config = _state["config"]

    cloud_only = _state.get("cloud_only_startup", False) or _state.get("defer_model", False)

    # Track why we're cloud-only
    if _state.get("cloud_only_startup", False):
        _state["cloud_only_reason"] = "explicit"
    elif _state.get("defer_model", False):
        _state["cloud_only_reason"] = "training"
    else:
        _state["cloud_only_reason"] = None

    # Auto-detect GPU conflict: if another process holds the GPU, start cloud-only
    if not cloud_only and _gpu_occupied():
        logger.warning(
            "GPU is occupied by another process — starting in cloud-only mode. "
            "Will auto-reclaim when GPU is free."
        )
        notify_server(SERVER_CLOUD_ONLY)
        cloud_only = True
        _state["cloud_only_reason"] = "gpu_conflict"

    if cloud_only:
        logger.info("Starting in cloud-only mode — skipping model load")
        _state["model"] = None
        _state["tokenizer"] = None
    else:
        logger.info("Loading model: %s (%s)", config.model_name, config.model_config.model_id)
        model, tokenizer = load_base_model(config.model_config)

        # Load enabled adapters that exist on disk
        for adapter_name, adapter_cfg in (
            ("episodic", config.adapters.episodic),
            ("semantic", config.adapters.semantic),
            ("procedural", config.adapters.procedural),
        ):
            if not adapter_cfg.enabled:
                continue
            adapter_path = config.adapter_dir / adapter_name
            if adapter_path.exists():
                logger.info("Loading adapter: %s from %s", adapter_name, adapter_path)
                model = load_adapter(model, str(config.adapter_dir), adapter_name)
        if hasattr(model, "peft_config") and model.peft_config:
            logger.info("Adapters loaded: %s", list(model.peft_config.keys()))
        else:
            logger.info("No adapters found — starting fresh")

        _state["model"] = model
        _state["tokenizer"] = tokenizer
    _state["session_buffer"] = SessionBuffer(
        config.session_dir,
        retain_sessions=config.consolidation.retain_sessions,
        debug=config.debug,
        snapshot_key=config.snapshot_key,
    )
    _state["session_buffer"].load_snapshot()

    # Initialize speaker store (voice-based identification)
    if config.speaker.enabled:
        from paramem.server.speaker import SpeakerStore

        speaker_path = (
            Path(config.speaker.store_path)
            if config.speaker.store_path
            else config.paths.data / "speaker_profiles.json"
        )
        _state["speaker_store"] = SpeakerStore(
            speaker_path,
            high_threshold=config.speaker.high_confidence_threshold,
            low_threshold=config.speaker.low_confidence_threshold,
            max_embeddings=config.speaker.max_embeddings_per_profile,
            redundancy_threshold=config.speaker.redundancy_threshold,
        )
        logger.info("Speaker store: %d profiles", _state["speaker_store"].profile_count)

        # Preload speaker embedding model (CPU, ~17 MB)
        from paramem.server.speaker_embedding import load_embedding_model

        if load_embedding_model():
            logger.info("Speaker embedding model ready")
        else:
            logger.warning("Speaker embedding unavailable — install paramem[speaker]")
    else:
        _state["speaker_store"] = None
        logger.info("Speaker identification disabled")

    # Initialize local STT (Faster Whisper) if configured
    # In cloud-only mode, force CPU to avoid GPU contention with training
    _state["stt"] = None
    _state["wyoming_server"] = None
    if config.stt.enabled:
        from paramem.server.stt import WhisperSTT

        if cloud_only and config.stt.device != "cpu":
            stt_device = "cpu"
            stt_model = config.stt.cpu_fallback_model
            stt_compute = "int8"
            logger.info(
                "Cloud-only mode — Whisper on CPU (%s instead of %s)",
                stt_model,
                config.stt.model,
            )
        else:
            stt_device = config.stt.device
            stt_model = config.stt.model
            stt_compute = config.stt.compute_type

        stt = WhisperSTT(
            model_name=stt_model,
            device=stt_device,
            compute_type=stt_compute,
            language=config.stt.language,
            beam_size=config.stt.beam_size,
            vad_filter=config.stt.vad_filter,
        )
        if stt.load():
            _state["stt"] = stt
            logger.info("Local STT: Whisper %s on %s", stt_model, stt_device)

            # Start Wyoming STT server
            from paramem.server.wyoming_handler import start_wyoming_server

            def _on_stt_embedding(embedding):
                """Store latest speaker embedding from Wyoming STT."""
                if embedding:
                    _state["latest_embedding"] = embedding

            def _on_stt_language(language: str, probability: float):
                """Store latest detected language from Wyoming STT.

                Written as single dict to avoid race between language and probability.
                Read by both /chat endpoint and TTS language resolver.
                """
                _state["latest_language_detection"] = {
                    "language": language,
                    "probability": probability,
                }

            _state["wyoming_server"] = await start_wyoming_server(
                host=config.server.host,
                port=config.stt.port,
                stt=stt,
                speaker_store=_state.get("speaker_store"),
                embedding_callback=_on_stt_embedding,
                language_callback=_on_stt_language,
            )
            logger.info("Wyoming STT server listening on port %d", config.stt.port)
        else:
            logger.warning("Local STT disabled — model failed to load")
    else:
        logger.info("Local STT: disabled")

    # Initialize local TTS if configured
    if config.tts.enabled:
        from paramem.server.tts import TTSManager

        tts_config = config.tts
        if cloud_only and tts_config.device != "cpu":
            logger.info("Cloud-only mode — forcing TTS to CPU")
            # Override device to CPU so engines don't attempt GPU loading
            tts_config = TTSConfig(
                enabled=tts_config.enabled,
                port=tts_config.port,
                device="cpu",
                default_language=tts_config.default_language,
                language_confidence_threshold=tts_config.language_confidence_threshold,
                model_dir=tts_config.model_dir,
                audio_chunk_bytes=tts_config.audio_chunk_bytes,
                voices=tts_config.voices,
            )

        tts_manager = TTSManager(
            tts_config,
            vram_safety_margin_mb=config.server.vram_safety_margin_mb,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts_manager.load_all)

        if tts_manager.is_loaded:
            _state["tts_manager"] = tts_manager
            logger.info("Local TTS: %s", ", ".join(tts_manager.available_languages))

            from paramem.server.wyoming_handler import start_wyoming_tts_server

            lang_conf_threshold = config.tts.language_confidence_threshold

            def _resolve_language():
                """Return the most recently detected language if confidence is sufficient.

                Consumes (pops) the detection so stale values don't persist
                across requests.
                """
                detection = _state.pop("latest_language_detection", None)
                if not detection:
                    return None
                if detection["probability"] >= lang_conf_threshold:
                    return detection["language"]
                return None

            _state["wyoming_tts_server"] = await start_wyoming_tts_server(
                host=config.server.host,
                port=config.tts.port,
                tts_manager=tts_manager,
                language_resolver=_resolve_language,
                audio_chunk_bytes=config.tts.audio_chunk_bytes,
            )
            logger.info("Wyoming TTS server listening on port %d", config.tts.port)
        else:
            logger.warning("Local TTS disabled — no voices loaded")
    else:
        logger.info("Local TTS: disabled")

    # Initialize cloud agent if configured
    _state["cloud_agent"] = get_cloud_agent(config.general_agent)
    if _state["cloud_agent"]:
        logger.info(
            "Cloud agent: %s (%s)",
            config.general_agent.provider,
            config.general_agent.model,
        )
    else:
        logger.info("Cloud agent: not configured (local-only mode)")

    # Initialize SOTA agent if configured
    _state["sota_agent"] = get_cloud_agent(config.sota_agent)
    if _state["sota_agent"]:
        logger.info(
            "SOTA agent: %s (%s)",
            config.sota_agent.provider,
            config.sota_agent.model,
        )
    else:
        logger.info("SOTA agent: not configured")

    # Register additional SOTA providers for direct routing (sota:anthropic, etc.)
    _state["sota_providers"] = {}
    for name, provider_config in config.sota_providers.items():
        agent = get_cloud_agent(provider_config)
        if agent:
            _state["sota_providers"][name] = agent
            logger.info("SOTA provider registered: %s (%s)", name, provider_config.model)
    logger.info("SOTA providers available: %s", list(_state["sota_providers"].keys()))

    # Initialize HA client, tool registry, and HA entity graph
    ha_graph = None
    tools_config = config.tools
    if tools_config.ha.url and tools_config.ha.token:
        ha_client = HAClient(
            url=tools_config.ha.url,
            token=tools_config.ha.token,
            timeout=tools_config.tool_timeout_seconds,
        )
        health = ha_client.health_check()
        if health:
            logger.info("HA client: connected to %s", tools_config.ha.url)
            entity_count = ha_client.load_entity_map()
            logger.info("HA entity map: %d entities", entity_count)

            # Build HA entity graph for dual-graph routing
            ha_services = ha_client.get_services()
            ha_graph = HAEntityGraph.build(ha_client._raw_states, ha_services)
        else:
            logger.warning("HA client: configured but unreachable at %s", tools_config.ha.url)
        _state["ha_client"] = ha_client

        # Load tool definitions
        registry = ToolRegistry()
        if tools_config.ha.auto_discover:
            ha_services = ha_client.get_services() if not ha_graph else ha_services
            if ha_services:
                registry.load_from_ha(
                    ha_services,
                    allowlist=tools_config.ha.allowlist,
                    sensitive_override=tools_config.ha.sensitive_override,
                )
        registry.load_from_yaml(tools_config.definitions)
        _state["tool_registry"] = registry
        logger.info("Tool registry: %d tools loaded", len(registry.tools))
    else:
        logger.info("HA tools: not configured")

    _state["ha_graph"] = ha_graph
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
        ha_graph=ha_graph,
    )

    # Start consolidation scheduler if configured
    if config.consolidation.schedule:
        _state["scheduler_task"] = asyncio.create_task(
            _consolidation_scheduler(config.consolidation.schedule)
        )
        logger.info("Consolidation scheduled at: %s", config.consolidation.schedule)

    # Start background training scheduler if configured
    training_interval = config.consolidation.training_interval_hours
    if training_interval > 0 and not cloud_only:
        _state["training_scheduler_task"] = asyncio.create_task(
            _training_scheduler(training_interval)
        )
        logger.info("Background training scheduled every %dh", training_interval)

    # Graceful shutdown: save encrypted session snapshot before exit.
    # SIGUSR1 (GPU release for training) and SIGTERM (systemd stop) both
    # trigger a snapshot so unconsolidated conversations survive restarts.
    def _graceful_exit(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — saving session snapshot before exit", sig_name)
        # Signal training to stop at the next epoch boundary
        consolidation_loop = _state.get("consolidation_loop")
        if consolidation_loop is not None:
            consolidation_loop.shutdown_requested = True
            logger.info("Shutdown flag set — training will stop after current epoch")
        bg_trainer = _state.get("background_trainer")
        if bg_trainer is not None and bg_trainer.is_training:
            bg_trainer.stop(timeout=30)
            logger.info("Background trainer stopped")
        buffer = _state.get("session_buffer")
        if buffer:
            buffer.save_snapshot()
        if signum == signal.SIGUSR1:
            # GPU release: exit non-zero so systemd restarts (Restart=on-failure)
            os._exit(1)
        else:
            # SIGTERM: intentional stop, don't restart
            raise SystemExit(0)

    signal.signal(signal.SIGUSR1, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    _state["mode"] = "cloud-only" if cloud_only else "local"
    _state["event_loop"] = asyncio.get_running_loop()

    # Auto-reclaim: only when started without the model (--defer-model).
    # In local mode we already have the GPU — nothing to reclaim.
    if cloud_only and not _state.get("cloud_only_startup", False):
        reclaim_interval = config.server.reclaim_interval_minutes
        _state["reclaim_task"] = asyncio.create_task(_auto_reclaim_loop(reclaim_interval))
    # Deferred enrollment: start idle loop (all state lives in RAM)
    if not cloud_only and config.speaker.enabled:
        idle_timeout = config.speaker.enrollment_idle_timeout
        _state["enrollment_task"] = asyncio.create_task(_enrollment_idle_loop(idle_timeout))

    logger.info("ParaMem server ready — mode: %s, model: %s", _state["mode"], config.model_name)

    yield

    # Shutdown — flush deferred speaker profile writes
    store = _state.get("speaker_store")
    if store:
        store.flush()

    if _state["scheduler_task"]:
        _state["scheduler_task"].cancel()
    if _state.get("training_scheduler_task"):
        _state["training_scheduler_task"].cancel()
    if _state["reclaim_task"]:
        _state["reclaim_task"].cancel()
    if _state.get("enrollment_task"):
        _state["enrollment_task"].cancel()
    wyoming_server = _state.get("wyoming_server")
    if wyoming_server is not None:
        wyoming_server.stop()
    wyoming_tts = _state.get("wyoming_tts_server")
    if wyoming_tts is not None:
        wyoming_tts.stop()
    stt = _state.get("stt")
    if stt is not None:
        stt.unload()
    tts_manager = _state.get("tts_manager")
    if tts_manager is not None:
        tts_manager.unload_all()
    if _state.get("ha_client"):
        _state["ha_client"].close()
    if _state["model"]:
        unload_model(_state["model"], _state["tokenizer"])
        logger.info("Model unloaded")


app = FastAPI(title="ParaMem", version="0.1.0", lifespan=lifespan)


# --- Endpoints ---


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a conversation turn with speaker identification."""
    _state["last_chat_time"] = datetime.now(timezone.utc)
    buffer = _state["session_buffer"]

    # Forced routing — bypass normal routing for direct provider testing.
    # Supports: "ha", "sota", "sota:anthropic", "sota:openai", "sota:google"
    if request.route and request.route.startswith(("ha", "sota")):
        _speaker_id, speaker = _resolve_speaker(request, buffer, _state.get("speaker_store"))
        loop = asyncio.get_running_loop()

        result = None
        if request.route == "ha" and _state.get("ha_client") is not None:
            response_text = await loop.run_in_executor(
                None,
                lambda: _state["ha_client"].conversation_process(
                    request.text, agent_id=_state["config"].ha_agent_id
                ),
            )
            if response_text is not None:
                result = ChatResult(text=response_text, escalated=True)
        elif request.route.startswith("sota"):
            parts = request.route.split(":", 1)
            agent = (
                _state.get("sota_providers", {}).get(parts[1])
                if len(parts) == 2
                else _state.get("sota_agent")
            )
            if agent is not None:
                result = await loop.run_in_executor(
                    None,
                    lambda: _escalate_to_sota(
                        request.text,
                        agent,
                        _state["config"],
                        speaker=speaker,
                        history=request.history,
                    ),
                )
        if result and result.text:
            return ChatResponse(text=result.text, escalated=True, speaker=speaker)
        return ChatResponse(
            text=f"Route '{request.route}' unavailable.",
            escalated=False,
            speaker=speaker,
        )

    # Pick up speaker embedding from latest STT if not in request
    latest_embedding = _state.get("latest_embedding")
    if not request.speaker_embedding and latest_embedding is not None:
        request.speaker_embedding = latest_embedding
        _state["latest_embedding"] = None
        logger.info("Picked up STT embedding (%d dims)", len(request.speaker_embedding))
    elif not request.speaker_embedding:
        logger.info("No speaker embedding available")

    # Pick up detected language from latest STT.
    # Not cleared here — TTS resolver reads it independently during synthesis
    # (which runs after /chat returns). Cleared after TTS consumes it via
    # the Wyoming handler's on_synthesize callback.
    lang_detection = _state.get("latest_language_detection")
    detected_language = lang_detection["language"] if lang_detection else None
    detected_language_prob = lang_detection["probability"] if lang_detection else 0.0
    if detected_language:
        logger.info("Detected language: %s (prob=%.2f)", detected_language, detected_language_prob)

    # Discard embeddings from short utterances — pyannote needs enough voice
    # data for a stable print. Short commands produce noisy, unreliable embeddings.
    min_words = _state["config"].speaker.min_embedding_words
    word_count = len(request.text.split()) if request.text else 0
    if request.speaker_embedding and word_count < min_words:
        logger.info(
            "Embedding discarded: transcript too short (%d words < %d minimum)",
            word_count,
            min_words,
        )
        request.speaker_embedding = None

    # Speaker resolution: embedding → session history → anonymous.
    # Never let speaker ID failure kill the request — proceed as anonymous.
    try:
        speaker_id, speaker = _resolve_speaker(request, buffer, _state.get("speaker_store"))
    except Exception:
        logger.exception("Speaker resolution failed — proceeding as anonymous")
        speaker_id, speaker = None, None
    follow_up = None
    store = _state.get("speaker_store")

    # Deferred enrollment: unknown voice → group by embedding silently.
    # Prompt only after global cooldown elapses — never on first encounter.
    # Enrollment failure must never block the query.
    try:
        if speaker_id is None and request.speaker_embedding and store:
            conv_id = request.conversation_id
            now = datetime.now(timezone.utc)
            unknown_group_id = _match_unknown_speaker(request.speaker_embedding)

            if unknown_group_id:
                group = _state["unknown_speakers"][unknown_group_id]
                group["conversations"].add(conv_id)
                group["embeddings"].append(request.speaker_embedding)
                _state["pending_enrollments"].add(conv_id)
                logger.info("Unknown speaker — grouped into %s", unknown_group_id)
            else:
                unknown_group_id = uuid.uuid4().hex[:8]
                _state["unknown_speakers"][unknown_group_id] = {
                    "embeddings": [request.speaker_embedding],
                    "conversations": {conv_id},
                    "first_seen": now,
                }
                _state["pending_enrollments"].add(conv_id)
                logger.info("Unknown speaker — new group %s", unknown_group_id)

            # Global enrollment prompt cooldown
            reprompt_interval = _state["config"].speaker.enrollment_reprompt_interval
            last_prompt = _state["last_enrollment_prompt"]
            if (now - last_prompt).total_seconds() >= reprompt_interval:
                follow_up = _state["config"].speaker.enrollment_prompt
                _state["last_enrollment_prompt"] = now
                logger.info("Enrollment prompt sent (cooldown %ds)", reprompt_interval)
    except Exception:
        logger.exception("Speaker enrollment failed — continuing without enrollment")

    # Update speaker language preference from STT detection
    tts_config = _state["config"].tts
    if speaker_id and store and detected_language and detected_language_prob > 0:
        store.update_language(
            speaker_id,
            detected_language,
            detected_language_prob,
            threshold=tts_config.language_confidence_threshold,
        )

    # Resolve effective language for this request:
    # 1. High-confidence Whisper detection
    # 2. Speaker's stored preference
    # 3. Config default (English)
    lang_threshold = tts_config.language_confidence_threshold
    if detected_language and detected_language_prob >= lang_threshold:
        effective_language = detected_language
    elif speaker_id and store:
        effective_language = store.get_preferred_language(speaker_id)
    else:
        effective_language = None
    # Replace detected_language with the resolved effective language
    detected_language = effective_language

    # Check greeting before routing (applies to all paths)
    greeting_prefix = None
    greeting_interval = _state["config"].voice.greeting_interval_hours
    if speaker and speaker_id and store and greeting_interval > 0:
        greeting = store.should_greet(speaker_id, greeting_interval)
        if greeting:
            greeting_prefix = f"{greeting}, {speaker}. "
            store.confirm_greeting(speaker_id)

    # Cloud-only mode — route via HA graph + SOTA, no local model
    if _state["mode"] == "cloud-only":
        result = _cloud_only_route(
            text=request.text,
            speaker=speaker,
            history=request.history,
            config=_state["config"],
            router=_state.get("router"),
            ha_client=_state.get("ha_client"),
            sota_agent=_state.get("sota_agent"),
            language=detected_language,
        )
        buffer.append(
            request.conversation_id,
            "user",
            request.text,
            embedding=request.speaker_embedding,
        )
        cloud_text = result.text
        buffer.append(request.conversation_id, "assistant", cloud_text)
        spoken_text = f"{greeting_prefix}{cloud_text}" if greeting_prefix else cloud_text
        return ChatResponse(text=spoken_text, escalated=True, speaker=speaker, follow_up=follow_up)

    # Local mode — normal inference with entity routing
    # Pause background training if active, then acquire GPU lock
    bg_trainer = _state.get("background_trainer")
    training_paused = False
    training_stopped = False
    if bg_trainer is not None and bg_trainer.is_training:
        if bg_trainer.pause():
            training_paused = True
        else:
            logger.warning("Could not pause training — stopping trainer before inference")
            bg_trainer.stop(timeout=30)
            training_stopped = True

    from paramem.server.wyoming_handler import gpu_inference_lock

    async with gpu_inference_lock:
        loop = asyncio.get_running_loop()
        result: ChatResult = await loop.run_in_executor(
            None,
            lambda: handle_chat(
                text=request.text,
                conversation_id=request.conversation_id,
                speaker=speaker,
                speaker_id=speaker_id,
                history=request.history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=_state["config"],
                router=_state["router"],
                cloud_agent=_state.get("cloud_agent"),
                sota_agent=_state.get("sota_agent"),
                ha_client=_state.get("ha_client"),
                tool_registry=_state.get("tool_registry"),
                language=detected_language,
            ),
        )

    buffer.append(
        request.conversation_id,
        "user",
        request.text,
        embedding=request.speaker_embedding,
    )
    response_text = result.text
    buffer.append(request.conversation_id, "assistant", response_text)

    # Resume training after inference — only if we paused (not stopped)
    if training_paused and bg_trainer is not None:
        bg_trainer.resume()
    elif training_stopped:
        logger.warning(
            "Training was stopped for inference — will restart on next scheduled training interval"
        )

    spoken_text = f"{greeting_prefix}{response_text}" if greeting_prefix else response_text
    return ChatResponse(
        text=spoken_text,
        escalated=result.escalated,
        speaker=speaker,
        follow_up=follow_up,
    )


def _match_unknown_speaker(embedding: list[float]) -> str | None:
    """Match an embedding against unknown speaker groups.

    Uses a lenient threshold (low_confidence * 0.6 ≈ 0.27) because we're
    grouping noisy unknowns, not confirming identity. Compares against each
    group's centroid (improves as more embeddings accumulate).
    """
    from paramem.server.speaker import compute_centroid, cosine_similarity

    cfg = _state["config"].speaker
    threshold = cfg.low_confidence_threshold * cfg.grouping_threshold_factor
    best_id = None
    best_score = 0.0
    for group_id, group in _state["unknown_speakers"].items():
        centroid = compute_centroid(group["embeddings"])
        if not centroid:
            continue
        score = cosine_similarity(embedding, centroid)
        if score >= threshold and score > best_score:
            best_score = score
            best_id = group_id
    return best_id


def _resolve_speaker(request: ChatRequest, buffer, speaker_store) -> tuple[str | None, str | None]:
    """Resolve speaker identity from multiple sources.

    Returns (speaker_id, speaker_name) tuple.

    Priority:
    1. Voice embedding match (via SpeakerStore, high confidence only)
    2. Session history (previously identified in this conversation)
    3. Anonymous (None, None)
    """
    # 1. Voice embedding match
    if request.speaker_embedding and speaker_store:
        match = speaker_store.match(request.speaker_embedding)
        if match.speaker_id and not match.tentative:
            buffer.set_speaker(request.conversation_id, match.speaker_id, match.name)
            # Enrich profile with this embedding (strengthens cross-device centroid)
            speaker_store.add_embedding(match.speaker_id, request.speaker_embedding)
            return match.speaker_id, match.name

    # 2. Previously identified in session
    existing_id = buffer.get_speaker_id(request.conversation_id)
    existing_name = buffer.get_speaker(request.conversation_id)
    if existing_id:
        return existing_id, existing_name

    return None, None


@app.get("/status", response_model=StatusResponse)
async def status():
    """Server health and state."""
    config = _state["config"]
    model = _state["model"]

    adapter_loaded = (
        hasattr(model, "peft_config") and "episodic" in model.peft_config if model else False
    )

    keys_count = 0
    if config.registry_path.exists():
        with open(config.registry_path) as f:
            registry = json.load(f)
        keys_count = len(registry)

    return StatusResponse(
        model=config.model_name,
        mode=_state["mode"],
        cloud_only_reason=_state.get("cloud_only_reason"),
        adapter_loaded=adapter_loaded,
        keys_count=keys_count,
        pending_sessions=_state["session_buffer"].pending_count if _state["session_buffer"] else 0,
        consolidating=_state["consolidating"],
        last_consolidation=_state["last_consolidation"],
        speaker_profiles=_state["speaker_store"].profile_count
        if _state.get("speaker_store")
        else 0,
        stt_loaded=_state.get("stt") is not None and _state["stt"].is_loaded,
        stt_model=_state["stt"].model_name if _state.get("stt") else None,
    )


@app.post("/refresh-ha")
async def refresh_ha():
    """Rebuild the HA entity graph from the HA API.

    Call after adding/removing devices, renaming entities, or
    reorganizing areas in Home Assistant.
    """
    ha_client = _state.get("ha_client")
    if ha_client is None:
        return {"status": "not_configured"}

    ha_client.load_entity_map()
    ha_services = ha_client.get_services()
    ha_graph = _state.get("ha_graph")
    if ha_graph is not None:
        ha_graph.refresh(ha_client._raw_states, ha_services)
    else:
        ha_graph = HAEntityGraph.build(ha_client._raw_states, ha_services)
        _state["ha_graph"] = ha_graph
        # Rebuild router with new graph
        config = _state["config"]
        _state["router"] = QueryRouter(
            adapter_dir=config.adapter_dir,
            ha_graph=ha_graph,
        )

    return {
        "status": "refreshed",
        "entities": ha_graph.entity_count,
        "areas": ha_graph.area_count,
        "verbs": ha_graph.verb_count,
    }


@app.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate():
    """Trigger consolidation manually."""
    if _state["mode"] == "cloud-only":
        return ConsolidateResponse(status="rejected_cloud_only")
    if _state["consolidating"]:
        return ConsolidateResponse(status="already_running")

    _state["consolidating"] = True
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _run_consolidation_sync)
    future.add_done_callback(_consolidation_done_callback)

    return ConsolidateResponse(status="started")


def _cloud_only_route(
    text: str,
    speaker: str | None,
    history: list[dict] | None,
    config,
    router=None,
    ha_client=None,
    sota_agent=None,
    language: str | None = None,
) -> ChatResult:
    """Route queries when the local model is unavailable (cloud-only mode).

    HA first (has tools for weather, time, devices), SOTA as fallback
    for reasoning. Mutual fallback: if one fails, try the other.
    """
    from paramem.server.sanitizer import sanitize_for_cloud

    # Cloud-only: no local model for PA probing.
    # HA first (has tools for weather, time, devices), SOTA as fallback.

    # Try HA conversation agent — it has tools and real-time data
    # Language passed via HA's native conversation API parameter
    if ha_client is not None:
        logger.debug("Cloud-only route: trying HA agent for: %s", text[:100])
        ha_languages = config.tools.ha.supported_languages
        response_text = ha_client.conversation_process(
            text,
            agent_id=config.ha_agent_id,
            language=language,
            supported_languages=ha_languages,
        )
        if response_text is not None:
            logger.info("Cloud-only route: HA agent responded")
            return ChatResult(text=response_text, escalated=True)
        logger.info("Cloud-only route: HA agent failed, trying SOTA")

    # HA failed or unavailable → try SOTA for reasoning
    if sota_agent is not None:
        sanitized, _ = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            logger.info("Cloud-only route: escalating to SOTA")
            result = _escalate_to_sota(
                sanitized,
                sota_agent,
                config,
                speaker=speaker,
                history=history,
                language=language,
            )
            if result.text:
                logger.info("Cloud-only route: SOTA responded")
                return result

    logger.warning("Cloud-only route: all services failed")
    return ChatResult(
        text="I'm running in limited mode right now. Please try again later.",
        escalated=True,
    )


# --- Internal ---


def _run_consolidation_sync():
    """Run consolidation in a background thread.

    The consolidating flag is set by the caller before submitting to
    the executor, and cleared by _consolidation_done_callback after
    completion. This eliminates the race window.
    """
    # Read HA home context for location validation
    ha_context = None
    ha_client = _state.get("ha_client")
    if ha_client is not None:
        ha_context = ha_client.get_home_context()

    result = run_consolidation(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        config=_state["config"],
        session_buffer=_state["session_buffer"],
        loop=_state["consolidation_loop"],
        ha_context=ha_context,
    )
    # Store loop for reuse across runs (preserves graph, registries, cycle count)
    if "loop" in result:
        loop = result.pop("loop")
        _state["consolidation_loop"] = loop
        # Update model reference — ConsolidationLoop wraps the base model
        # in PeftModel with adapters during training
        _state["model"] = loop.model
    _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    _state["router"].reload()
    # Refresh HA entity graph if available
    ha_graph = _state.get("ha_graph")
    ha_client = _state.get("ha_client")
    if ha_graph is not None and ha_client is not None:
        ha_client.load_entity_map()
        ha_services = ha_client.get_services()
        ha_graph.refresh(ha_client._raw_states, ha_services)
    logger.info("Consolidation result: %s", result)


def _consolidation_done_callback(future):
    """Called when consolidation completes or fails."""
    _state["consolidating"] = False
    exc = future.exception()
    if exc:
        logger.exception("Consolidation failed: %s", exc)


async def _consolidation_scheduler(schedule: str):
    """Background task that triggers consolidation at the configured hour.

    Schedule format: "HH:MM" (24-hour).
    """
    try:
        target_hour, target_minute = map(int, schedule.split(":"))
    except ValueError:
        logger.error("Invalid schedule format: %s (expected HH:MM)", schedule)
        return

    triggered_today = False

    while True:
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute

        if current_hour == target_hour and current_minute == target_minute:
            if not triggered_today and not _state["consolidating"] and _state["mode"] == "local":
                logger.info("Scheduled consolidation triggered at %s", schedule)
                _state["consolidating"] = True
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, _run_consolidation_sync)
                future.add_done_callback(_consolidation_done_callback)
                triggered_today = True
        else:
            triggered_today = False

        await asyncio.sleep(30)


# --- Background training scheduler ---


async def _training_scheduler(interval_hours: int):
    """Extract pending sessions and start background training.

    Every N hours:
    1. Extract all pending sessions (blocking — needs model for LLM extraction)
    2. Prepare training data (key assignment, reconstruction)
    3. Start BackgroundTrainer (non-blocking — trains in daemon thread)

    Skips if no pending sessions, already consolidating/training, or cloud-only.
    """
    interval_seconds = interval_hours * 3600

    while True:
        await asyncio.sleep(interval_seconds)

        if _state["consolidating"]:
            logger.info("Training scheduler: consolidation already running, skipping")
            continue
        if _state["mode"] != "local":
            continue
        bg = _state.get("background_trainer")
        if bg is not None and bg.is_training:
            logger.info("Training scheduler: background training active, skipping")
            continue

        pending = _state["session_buffer"].get_pending()
        if not pending:
            logger.info("Training scheduler: no pending sessions, skipping")
            continue

        has_speaker = any(s.get("speaker_id") for s in pending)
        if not has_speaker:
            logger.info("Training scheduler: no sessions with speaker_id, skipping")
            continue

        logger.info(
            "Training scheduler: %d pending sessions, starting extract + train",
            len(pending),
        )
        _state["consolidating"] = True

        # Fire-and-forget — extraction runs in executor, training in daemon thread
        event_loop = asyncio.get_running_loop()
        future = event_loop.run_in_executor(None, _extract_and_start_training)
        future.add_done_callback(_training_scheduler_done_callback)


def _training_scheduler_done_callback(future):
    """Called when extraction phase completes (training continues in background)."""
    exc = future.exception()
    if exc:
        logger.exception("Training scheduler extraction failed: %s", exc)
        _state["consolidating"] = False


def _extract_and_start_training():
    """Extract sessions, prepare training data, start background trainer.

    Runs in executor thread. Extraction is blocking (LLM inference).
    Training starts in a background daemon thread and returns immediately.
    """
    from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
    from paramem.server.consolidation import (
        _increment_key_sessions,
        _promote_mature_keys,
        _save_key_metadata,
        _save_keyed_pairs_for_router,
        _save_registry,
        _save_simulation_results,
        create_consolidation_loop,
    )

    config = _state["config"]
    session_buffer = _state["session_buffer"]

    # Create or reuse consolidation loop
    loop = _state.get("consolidation_loop")
    if loop is None:
        loop = create_consolidation_loop(_state["model"], _state["tokenizer"], config)
        _state["consolidation_loop"] = loop
        _state["model"] = loop.model

    # Read HA home context for location validation
    ha_context = None
    ha_client = _state.get("ha_client")
    if ha_client is not None:
        ha_context = ha_client.get_home_context()

    # --- Phase 1: Extract all sessions ---
    pending = session_buffer.get_pending()
    all_episodic_qa = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []

    for session in pending:
        session_id = session["session_id"]
        transcript = session["transcript"]
        session_speaker_id = session.get("speaker_id")
        session_ids.append(session_id)

        if not session_speaker_id:
            continue

        if loop.shutdown_requested:
            logger.info("Shutdown — stopping extraction early")
            break

        episodic_qa, procedural_rels = loop.extract_session(
            transcript,
            session_id,
            speaker_id=session_speaker_id,
            ha_context=ha_context,
            stt_correction=config.consolidation.extraction_stt_correction,
            ha_validation=config.consolidation.extraction_ha_validation,
            noise_filter=config.consolidation.extraction_noise_filter,
            noise_filter_model=config.consolidation.extraction_noise_filter_model,
        )
        _increment_key_sessions(loop, session_id)

        for qa in episodic_qa:
            qa["speaker_id"] = session_speaker_id
        for rel in procedural_rels:
            rel["speaker_id"] = session_speaker_id

        all_episodic_qa.extend(episodic_qa)
        all_procedural_rels.extend(procedural_rels)
        speaker_ids.append(session_speaker_id)

    if not all_episodic_qa and not all_procedural_rels:
        logger.info("No QA pairs extracted — skipping")
        session_buffer.mark_consolidated(session_ids)
        _state["consolidating"] = False
        return

    # --- Simulate mode: save results, skip training ---
    if config.consolidation.mode == "simulate":
        if config.debug:
            _save_simulation_results(all_episodic_qa, all_procedural_rels, loop, config)
        session_buffer.mark_consolidated(session_ids)
        _state["consolidating"] = False
        logger.info(
            "Simulation complete: %d episodic QA, %d procedural rels",
            len(all_episodic_qa),
            len(all_procedural_rels),
        )
        return

    # --- Phase 2: Prepare training data (key assignment, reconstruction) ---
    if not loop.config.indexed_key_replay_enabled:
        logger.warning("Indexed key replay disabled — skipping training")
        session_buffer.mark_consolidated(session_ids)
        _state["consolidating"] = False
        return

    primary_speaker = speaker_ids[-1] if speaker_ids else ""
    jobs_data = loop.prepare_training_data(
        all_episodic_qa, all_procedural_rels, speaker_id=primary_speaker
    )

    if not jobs_data:
        logger.info("No training jobs prepared — skipping")
        session_buffer.mark_consolidated(session_ids)
        _state["consolidating"] = False
        return

    # Build TrainingJob objects
    adapter_configs = {
        "episodic": config.episodic_adapter_config,
        "semantic": config.semantic_adapter_config,
        "procedural": config.procedural_adapter_config,
    }
    training_jobs = [
        TrainingJob(
            keyed_pairs=keyed_pairs,
            adapter_name=adapter_name,
            adapter_config=adapter_configs[adapter_name],
        )
        for adapter_name, keyed_pairs in jobs_data
    ]

    # --- Phase 3: Start background training ---
    def _on_training_complete():
        """Called from training thread when all jobs finish.

        Disk I/O is safe from any thread. SimHash updates and state
        mutations are posted to the event loop via call_soon_threadsafe
        to avoid race conditions with inference reads.
        """
        # Disk I/O — safe from any thread
        loop.finalize_training()
        _promote_mature_keys(loop, config)
        _save_keyed_pairs_for_router(loop, config)
        _save_registry(loop, config)
        _save_key_metadata(loop, config)
        session_buffer.mark_consolidated(session_ids)

        # SimHash updates + state mutations + router reload — all on event loop
        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_background_training, loop, jobs_data)
        else:
            _finalize_background_training(loop, jobs_data)

    # Build TrainingJob objects and start background trainer
    bt = BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
    )

    def _on_training_error():
        loop.rollback_preparation()
        _state["consolidating"] = False
        logger.error("Background training failed — state rolled back")

    _state["background_trainer"] = bt
    bt.start_jobs(
        training_jobs,
        on_complete=_on_training_complete,
        on_error=_on_training_error,
    )
    logger.info(
        "Extraction done — %d training jobs started in background",
        len(training_jobs),
    )


def _finalize_background_training(loop, jobs_data=None):
    """Update shared server state after background training.

    Must run on the event loop thread to avoid racing with /chat.
    SimHash updates happen here (not in the training thread) for thread safety.
    """
    from paramem.training.indexed_memory import build_registry

    # Update SimHash registries on the event loop thread
    if jobs_data is not None:
        for adapter_name, keyed_pairs in jobs_data:
            new_registry = build_registry(keyed_pairs)
            if adapter_name == "episodic":
                loop.episodic_simhash.update(new_registry)
            elif adapter_name == "semantic":
                loop.semantic_simhash.update(new_registry)
            elif adapter_name == "procedural":
                loop.procedural_simhash.update(new_registry)

    loop.model.eval()
    _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    _state["router"].reload()
    _state["consolidating"] = False
    total_keys = (
        len(loop.episodic_simhash) + len(loop.semantic_simhash) + len(loop.procedural_simhash)
    )
    logger.info("Background training complete — %d keys saved", total_keys)


# --- Deferred speaker enrollment ---
#
# When an unknown speaker talks, their voice embedding is stored alongside
# each transcript turn. After an idle timeout (no /chat requests), the local
# model extracts the speaker's self-introduced name from the transcript.
# This replaces fragile regex-based name extraction with LLM reasoning.
#


def _extract_name_via_llm(
    turns: list[dict],
    model,
    tokenizer,
) -> str | None:
    """Use the local LLM to extract a speaker's self-introduced name from transcript.

    The LLM reasons about context to distinguish the speaker's own name
    from other names mentioned in conversation. Returns the name or None.
    """
    from paramem.evaluation.recall import generate_answer

    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        text = turn.get("text", "")
        lines.append(f"{role}: {text}")
    transcript_text = "\n".join(lines)

    system_msg = (
        "You extract speaker names from conversation transcripts. "
        "Return ONLY the first name the speaker claims as their own identity, "
        "or NONE if they did not introduce themselves. "
        "Do NOT extract names of other people mentioned in conversation."
    )
    user_msg = (
        "Extract the speaker's self-introduced name from this transcript.\n\n"
        "Examples:\n"
        'Transcript: "user: My name is Alex. What time is it?"\n'
        "Answer: Alex\n\n"
        'Transcript: "user: Tell me about John Smith\'s schedule"\n'
        "Answer: NONE\n\n"
        'Transcript: "user: Stop playing music"\n'
        "Answer: NONE\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    result = generate_answer(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0)
    result = result.strip().strip('"').strip("'").strip(".")

    if not result or result.upper() == "NONE" or len(result) > 30:
        return None

    words = result.split()
    if len(words) > 3 or len(words) == 0:
        return None

    return result


async def _enrollment_idle_loop(idle_timeout_seconds: int):
    """Background task: extract speaker names via LLM after idle timeout.

    Only runs when mode == 'local' (GPU available). Check interval configurable.
    Aborts the current batch if a new /chat request arrives.
    On cancellation (shutdown), flushes any pending speaker profile writes.
    """
    check_interval = _state["config"].speaker.enrollment_check_interval

    try:
        await _enrollment_idle_loop_inner(idle_timeout_seconds, check_interval)
    except asyncio.CancelledError:
        store = _state.get("speaker_store")
        if store:
            store.flush()
        raise


async def _enrollment_idle_loop_inner(idle_timeout_seconds: int, check_interval: int):
    """Inner loop — separated so the outer wrapper can catch CancelledError."""
    while True:
        await asyncio.sleep(check_interval)

        if _state["mode"] != "local":
            continue
        if not _state.get("pending_enrollments"):
            continue
        last_chat = _state.get("last_chat_time")
        if not last_chat:
            continue
        elapsed = (datetime.now(timezone.utc) - last_chat).total_seconds()
        if elapsed < idle_timeout_seconds:
            continue

        buffer = _state["session_buffer"]
        store = _state.get("speaker_store")
        model = _state.get("model")
        tokenizer = _state.get("tokenizer")

        if not store or not model or not tokenizer:
            continue

        from paramem.server.wyoming_handler import gpu_inference_lock

        # Process each unknown speaker group
        group_ids = list(_state["unknown_speakers"].keys())
        for group_id in group_ids:
            # Abort if new activity arrived
            new_last = _state.get("last_chat_time")
            if new_last and new_last != last_chat:
                logger.info("Enrollment deferred: new /chat activity detected")
                break

            group = _state["unknown_speakers"].get(group_id)
            if not group:
                continue

            # Collect all turns across this group's conversations
            all_turns = []
            for conv_id in group["conversations"]:
                all_turns.extend(buffer.get_session_turns(conv_id))

            if not all_turns:
                _state["pending_enrollments"] -= group["conversations"]
                del _state["unknown_speakers"][group_id]
                continue

            async with gpu_inference_lock:
                loop = asyncio.get_running_loop()
                extracted = await loop.run_in_executor(
                    None,
                    lambda t=all_turns: _extract_name_via_llm(t, model, tokenizer),
                )

            if extracted:
                from paramem.server.speaker import compute_centroid

                ref_embedding = compute_centroid(group["embeddings"])
                new_id = store.enroll(extracted, ref_embedding)
                if new_id:
                    # Attribute all grouped conversations
                    for conv_id in group["conversations"]:
                        buffer.set_speaker(conv_id, new_id, extracted)
                    claimed = buffer.claim_sessions_for_speaker(new_id, extracted, store)
                    logger.info(
                        "Deferred enrollment: %s (id=%s, group %s), claimed %d sessions",
                        extracted,
                        new_id,
                        group_id,
                        claimed,
                    )
                else:
                    logger.info(
                        "Deferred enrollment skipped for group %s (voice already enrolled)",
                        group_id,
                    )
            else:
                logger.info("No self-introduction found in group %s", group_id)

            _state["pending_enrollments"] -= group["conversations"]
            del _state["unknown_speakers"][group_id]


# --- GPU lifecycle ---
#
# GPU management is service-level: stop the service to free GPU,
# restart to reclaim. No in-process GPU release/reclaim — process
# exit is the only way to fully free the CUDA context.
#
# Flow:
#   tresume: stop service → start with --defer-model → launch training
#   Training finishes: auto-reclaim detects GPU free → restart service
#   Fresh lifespan loads both LLM + STT on startup


def _gpu_occupied() -> bool:
    """Check if another process is using the GPU at startup time.

    Used to prevent loading the model when an ML workload is running.
    """
    return _gpu_has_compute_processes()


def _gpu_has_compute_processes() -> bool:
    """Check if any non-server process is using the GPU."""
    try:
        result = subprocess.run(
            [_NVIDIA_SMI, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        # Filter out our own process
        own_pid = os.getpid()
        external_pids = [p for p in pids if p != own_pid]
        return len(external_pids) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        # Fail safe: if we can't check, assume GPU is occupied
        logger.warning(
            "nvidia-smi unavailable at %s (%s) — assuming GPU occupied",
            _NVIDIA_SMI,
            type(e).__name__,
        )
        return True


async def _auto_reclaim_loop(interval_minutes: int = 10):
    """Periodically check if GPU is free and reclaim it.

    Only started when the server is in cloud-only/defer mode (no model loaded).
    Checks every interval_minutes whether any GPU compute process is still
    running. If not, restarts the service to get a clean CUDA context with
    the model loaded.
    """
    interval_seconds = interval_minutes * 60
    while True:
        await asyncio.sleep(interval_seconds)
        if _gpu_has_compute_processes():
            logger.debug("Auto-reclaim: GPU still occupied, waiting")
            continue
        logger.info("Auto-reclaim: GPU free — restarting for clean model load")
        _restart_service()


def _restart_service():
    """Restart the systemd service for a clean process.

    Used by auto-reclaim: model loading requires CUDA initialization on the
    main thread at process startup. In-process reclaim from a thread pool
    creates meta-device tensors that aren't resident in VRAM.
    """
    logger.info("Restarting paramem-server service...")
    try:
        subprocess.Popen(
            ["systemctl", "--user", "restart", "paramem-server"],
            start_new_session=True,
        )
    except Exception:
        logger.exception("Failed to restart service")


# --- Entry point ---


def main():
    parser = argparse.ArgumentParser(description="ParaMem Server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/server.yaml",
        help="Path to server config YAML",
    )
    parser.add_argument(
        "--cloud-only",
        action="store_true",
        help="Start in cloud-only mode permanently (skip model loading, no auto-reclaim)",
    )
    parser.add_argument(
        "--defer-model",
        action="store_true",
        help="Start without model (cloud-only) but auto-reclaim GPU when free",
    )
    args = parser.parse_args()

    # Setup
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = load_server_config(args.config)
    _state["config"] = config
    _state["cloud_only_startup"] = args.cloud_only
    _state["defer_model"] = args.defer_model

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
