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
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from paramem.models.loader import load_adapter, load_base_model, switch_adapter, unload_model
from paramem.server.cloud import get_cloud_agent
from paramem.server.config import TTSConfig, load_server_config
from paramem.server.consolidation import run_consolidation
from paramem.server.ha_graph import HAEntityGraph
from paramem.server.inference import (
    ChatResult,
    _escalate_to_sota,
    enqueue_post_session_train,
    handle_chat,
)
from paramem.server.post_session_queue import PostSessionQueue
from paramem.server.router import QueryRouter
from paramem.server.session_buffer import SessionBuffer  # "session" here = conversation
from paramem.server.tools.ha_client import HAClient
from paramem.server.vram_validator import (
    ConfigurationError,
    assess_topology,
    enforce_live_budget,
    estimate_stt_bytes,
    estimate_tts_bytes,
    format_tier_table,
    measure_external_vram,
)
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
    "sota_agent": None,
    "sota_providers": {},
    "ha_client": None,
    "consolidation_loop": None,
    "consolidating": False,
    "last_consolidation": None,
    "background_trainer": None,
    "reclaim_task": None,
    "post_session_queue": None,  # PostSessionQueue instance (local mode only)
    "mode": "local",  # "local" or "cloud-only"
    "cloud_only_reason": None,  # "explicit", "training", "gpu_conflict", or None
    "cloud_only_startup": False,  # set by --cloud-only CLI flag before app start
    "defer_model": False,  # set by --defer-model CLI flag before app start
    "ha_graph": None,  # HAEntityGraph built from HA states/services at startup
    "event_loop": None,  # asyncio event loop reference for cross-thread scheduling
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
    # Full HF model identifier (e.g. "mistralai/Mistral-7B-Instruct-v0.3").
    # Lets pstatus surface the variant alongside the short name.
    model_id: str | None = None
    model_device: str | None = None  # cuda / cpu / None (cloud-only)
    mode: str  # "local" or "cloud-only"
    cloud_only_reason: str | None  # "explicit", "training", "gpu_conflict", or None
    adapter_loaded: bool  # legacy: True when episodic main adapter is loaded
    # Rank of the episodic LoRA adapter (load-bearing for indexed-key recall).
    # None when no adapter is configured (cloud-only or all kinds disabled).
    episodic_rank: int | None = None
    # Per-kind adapter spec. Episodic / semantic / procedural can diverge in
    # learning rate and target modules (procedural adds MLP targets for
    # representational imprint), so each kind gets its own row. Shape:
    #   {kind: {"rank", "alpha", "learning_rate", "target_kind"}}
    # "target_kind" is "attn" when target_modules are attention-only, or
    # "attn+mlp" when MLP layers are included.
    adapter_specs: dict = {}
    # Adapter inventory: kind → configured count (kinds absent with count==0).
    # Main kinds (episodic/semantic/procedural) contribute 1 when enabled in
    # yaml; "interim" contributes max_interim_count.
    adapter_config: dict[str, int] = {}
    # Name of the currently active adapter on the live PeftModel, or None
    # when no adapters are loaded (fresh install / cloud-only).
    active_adapter: str | None = None
    keys_count: int
    pending_sessions: int
    consolidating: bool
    last_consolidation: str | None
    speaker_profiles: int = 0
    # Speaker-embedding backend (pyannote) + HF model id + device. None in
    # three cases: speaker id disabled in yaml, pyannote not installed, or
    # the model failed to load at startup.
    speaker_embedding_backend: str | None = None
    speaker_embedding_model: str | None = None
    speaker_embedding_device: str | None = None
    stt_loaded: bool = False
    stt_model: str | None = None
    stt_device: str | None = None  # cuda / cpu / None (unloaded)
    stt_engine: str | None = None  # "whisper" — backend family, currently fixed
    tts_loaded: bool = False
    tts_languages: list[str] = []  # loaded TTS voices by language code
    tts_device: str | None = None  # cuda / cpu / mixed / None (unloaded)
    # Backend family across loaded voices: "piper", "mms_tts", or "piper+mms"
    # when voices span both. None when no TTS is loaded.
    tts_engine: str | None = None
    # Interim cadence knob + derived full-cycle period.
    refresh_cadence: str = ""
    consolidation_period: str = ""  # refresh_cadence × max_interim_count
    max_interim_count: int = 0
    mode_config: str = ""  # "train" or "simulate"
    next_run_seconds: int | None = None  # seconds until next FULL consolidation
    # Seconds until the next interim cadence boundary (post_session_train
    # rolls over to a new stamp at every such boundary). None when refresh
    # cadence is disabled.
    next_interim_seconds: int | None = None
    orphaned_pending: int = 0  # pending sessions without speaker_id
    oldest_pending_seconds: int | None = None
    speakers: list[dict] = []  # [{id, name, embeddings, pending, enroll_method}]
    bg_trainer_active: bool = False
    bg_trainer_adapter: str | None = None
    last_consolidation_result: dict | None = None  # last completed run summary
    pending_enrollments: int = 0  # unknown speakers awaiting name extraction
    scheduler_started: bool = False  # True once scheduler first ticked
    # adapter_id -> {status, reason, updated_at, keys_at_mark}
    adapter_health: dict = {}
    # Thermal-throttle / quiet-hours policy snapshot.
    # mode: "always_on" | "always_off" | "auto"
    # start/end: "HH:MM" local (populated for all modes, consumed only when mode=auto)
    # currently_throttling: true iff the thermal throttle is active right now
    thermal_policy: dict = {}


class ConsolidateResponse(BaseModel):
    status: str


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    config = _state["config"]

    # cloud_only is enabled if ANY of the following is true:
    #   1. --cloud-only CLI flag was passed at startup.
    #   2. cloud_only: true is set in server.yaml (YAML cannot be silently overridden).
    #   3. --defer-model flag was passed (start cloud-only then auto-reclaim GPU).
    # OR is the correct combiner: both are opt-in signals for cloud-only mode.
    cloud_only = (
        _state.get("cloud_only_startup", False)
        or config.cloud_only
        or _state.get("defer_model", False)
    )

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
        # Startup VRAM validation runs BEFORE loading the base model.
        # On (re)start ParaMem's own footprint is wiped — the CUDA context
        # begins empty for this process. ``torch.cuda.memory_allocated`` at
        # this point therefore reports bytes held by OTHER GPU consumers
        # (Windows desktop, another model server, etc.). That external
        # occupancy is subtracted from the hardware cap to yield the live
        # budget the configured topology must fit into. The topology math
        # itself is hardware-agnostic — logged at INFO with a per-tier fit
        # table so operators see what a bigger card would buy.
        main_adapter_count = sum(
            1
            for adapter_cfg in (
                config.adapters.episodic,
                config.adapters.semantic,
                config.adapters.procedural,
            )
            if adapter_cfg.enabled
        )
        stt_pre_bytes = estimate_stt_bytes(config.stt, cloud_only=cloud_only)
        tts_pre_bytes = estimate_tts_bytes(config.tts, cloud_only=cloud_only)

        if not torch.cuda.is_available():
            logger.error(
                "Local model mode requires a CUDA-capable GPU but none was detected. "
                "Either provide a GPU, or start in cloud-only mode."
            )
            sys.exit(1)

        assessment = assess_topology(
            config.episodic_adapter_config,
            max_interim_count=config.consolidation.max_interim_count,
            model_name=config.model_name,
            model_id=config.model_config.model_id,
            main_adapter_count=main_adapter_count,
            stt_bytes=stt_pre_bytes,
            tts_bytes=tts_pre_bytes,
        )
        logger.info("VRAM topology assessment:\n%s", assessment.breakdown)
        logger.info("%s", format_tier_table(assessment))

        total_memory_bytes, external_bytes = measure_external_vram()
        try:
            enforce_live_budget(assessment, total_memory_bytes, external_bytes)
        except ConfigurationError as exc:
            logger.error("VRAM configuration error:\n%s", exc)
            sys.exit(1)

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

        # Load interim adapters that survived from a previous run (e.g. after a
        # restart before the weekly consolidation ran).  These live at
        # adapter_dir/episodic_interim_YYYYMMDDTHHMM/ alongside the main adapters.
        # PEFT crashes if adapter_config.json exists without adapter_model.safetensors
        # (CLAUDE.md), so we validate both files before calling load_adapter.
        for _interim_path in sorted(config.adapter_dir.glob("episodic_interim_*")):
            if not _interim_path.is_dir():
                continue
            if (
                not (_interim_path / "adapter_config.json").exists()
                or not (_interim_path / "adapter_model.safetensors").exists()
            ):
                logger.warning("Skipping half-present interim adapter: %s", _interim_path.name)
                continue
            _interim_name = _interim_path.name
            model = load_adapter(model, str(config.adapter_dir), _interim_name)
            logger.info("Loaded interim adapter: %s", _interim_name)

        # Restore the main episodic adapter as the active adapter so that all
        # inference probes default to the consolidated tier.  Only needed when at
        # least one adapter is present; on a fresh install peft_config is absent.
        if hasattr(model, "peft_config") and "episodic" in model.peft_config:
            switch_adapter(model, "episodic")

        # I5 — Registry consistency check on startup.
        # If the process was killed between adapter-save and registry-save in
        # post_session_train, the registry may contain entries whose adapter
        # safetensors file is missing.  Drop such entries and log a warning so
        # the adapter is rebuilt at the next post-session training pass.
        _registry_path = config.adapter_dir / "indexed_key_registry.json"
        if _registry_path.exists():
            from paramem.training.key_registry import KeyRegistry as _KeyRegistry

            _reg = _KeyRegistry.load(_registry_path)
            _orphaned: list[str] = []
            for _key in list(_reg.list_active()):
                _aid = _reg.get_adapter_id(_key)
                # Only interim adapters need this check — main adapters are
                # saved by _save_adapters() which is always a complete write.
                # Interim safetensors live at adapter_dir/<adapter_id>/adapter_model.safetensors
                if _aid.startswith("episodic_interim_"):
                    _weights = config.adapter_dir / _aid / "adapter_model.safetensors"
                    if not _weights.exists():
                        _reg.remove(_key)
                        _orphaned.append(_key)
            if _orphaned:
                logger.warning(
                    "Startup registry check: dropped %d orphan key(s) whose adapter "
                    "weights are missing (adapter save was interrupted): %s",
                    len(_orphaned),
                    _orphaned,
                )
                _reg.save(_registry_path)

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
                min_embedding_duration_seconds=config.speaker.min_embedding_duration_seconds,
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

    # Post-load sanity: compare actual ParaMem allocation against the prediction.
    # Not a gate — the pre-load check already enforced the budget against
    # (total_memory − external_bytes). This surfaces any drift between the
    # static VRAM tables (base model, STT, TTS) and reality, so calibration
    # errors show up in logs rather than silently over-committing.
    if _state.get("model") is not None and torch.cuda.is_available():
        actual_bytes = torch.cuda.memory_allocated(0)
        predicted_bytes = assessment.required_bytes
        delta_mib = (actual_bytes - predicted_bytes) / (1024 * 1024)
        logger.info(
            "VRAM post-load sanity: predicted %.2f GiB, measured %.2f GiB (delta %+.0f MiB)",
            predicted_bytes / 2**30,
            actual_bytes / 2**30,
            delta_mib,
        )
        if actual_bytes > predicted_bytes * 1.10:
            logger.warning(
                "Actual VRAM usage exceeds prediction by >10%% — "
                "re-calibrate _MODEL_VRAM_BYTES / STT / TTS tables."
            )

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

    else:
        logger.info("HA tools: not configured")

    _state["ha_graph"] = ha_graph
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
        ha_graph=ha_graph,
    )

    # Global observed-language tracker — records STT-detected languages with
    # high confidence, publishes to HA as input_text.voice_observed_languages
    # so the conversation agent can use it as context when interpreting
    # (potentially mangled) transcripts from CPU fallback STT.
    from paramem.server.language_tracker import LanguageTracker

    _state["language_tracker"] = LanguageTracker(
        store_path=config.paths.data / "observed_languages.json",
        ha_client=_state.get("ha_client"),
    )

    # Reconcile the systemd user timer with the DERIVED full consolidation
    # period (= refresh_cadence × max_interim_count). The yaml exposes only
    # refresh_cadence; the timer sees the full cycle via the derived
    # ``consolidation_period_string`` property. Timer fires POST /scheduled-tick
    # on that cadence regardless of mode; the tick handler defers when GPU is
    # busy rather than skipping silently. Timer survives server restart.
    from paramem.server import systemd_timer

    derived_period = config.consolidation.consolidation_period_string
    logger.info(
        "Consolidation cadence — refresh every %s, max_interim_count=%d, "
        "derived full-consolidation period=%s",
        config.consolidation.refresh_cadence or "<disabled>",
        config.consolidation.max_interim_count,
        derived_period or "<manual only>",
    )
    try:
        msg = systemd_timer.reconcile(derived_period)
        logger.info("%s", msg)
    except Exception:
        logger.exception("Failed to reconcile consolidation timer — continuing without schedule")

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

    # Startup diagnostic: surface any in-flight training state left by a
    # prior interrupted run.  The actual resume happens when the next
    # training job for that adapter is submitted — this is informational only.
    if _state.get("model") is not None:
        from paramem.server.background_trainer import _RESUME_STATE_FILE, _read_resume_state

        _resume_path = config.adapter_dir / "in_training" / _RESUME_STATE_FILE
        _resume_state = _read_resume_state(_resume_path)
        if _resume_state is not None and _resume_state.get("checkpoint_path"):
            logger.info(
                "Detected in-flight training state: adapter=%s epoch=%d/%d"
                " — will resume when the job is next submitted",
                _resume_state.get("adapter_name", "?"),
                _resume_state.get("last_completed_epoch", 0),
                _resume_state.get("total_epochs", 0),
            )

    # --- Post-session queue: persistent queue for missed training triggers ---
    # Instantiate the queue in local mode so the chat handler can use it.
    # In cloud-only mode there is no model to train — skip.
    if not cloud_only:
        _queue_path = config.adapter_dir / "post_session_queue.json"
        # Ensure the adapter directory exists (may not exist on fresh install).
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        _state["post_session_queue"] = PostSessionQueue(_queue_path)

        # Replay any entries that were enqueued before a previous restart.
        # We need the ConsolidationLoop + BackgroundTrainer to replay — create
        # them lazily here only when there are entries to process.
        if config.consolidation.post_session_train_enabled:
            _pending = _state["post_session_queue"].peek()
            if _pending:
                logger.info(
                    "post_session_queue: %d pending entry(s) from previous run — replaying",
                    len(_pending),
                )
                from paramem.server.background_trainer import BackgroundTrainer
                from paramem.server.consolidation import create_consolidation_loop

                if _state.get("consolidation_loop") is None:
                    _replay_loop = create_consolidation_loop(
                        _state["model"], _state["tokenizer"], config
                    )
                    _state["consolidation_loop"] = _replay_loop
                    _state["model"] = _replay_loop.model

                if _state.get("background_trainer") is None:
                    _replay_bt = BackgroundTrainer(
                        model=_state["model"],
                        tokenizer=_state["tokenizer"],
                        training_config=config.training_config,
                        output_dir=config.adapter_dir,
                        temp_limit=config.consolidation.training_temp_limit,
                        temp_check_interval=config.consolidation.training_temp_check_interval,
                        quiet_hours_mode=config.consolidation.quiet_hours_mode,
                        quiet_hours_start=config.consolidation.quiet_hours_start,
                        quiet_hours_end=config.consolidation.quiet_hours_end,
                    )
                    _state["background_trainer"] = _replay_bt

                # Drain and replay — entries are removed individually upon
                # successful completion inside enqueue_post_session_train's
                # wrapper (same pattern as the chat handler path).
                _replay_entries = _state["post_session_queue"].drain()
                for _entry in _replay_entries:
                    # Re-enqueue via the same path as the chat handler uses.
                    # The queue.remove() call on success is handled inside the
                    # _run_post_session_train wrapper added below for the chat
                    # handler path.  For replay we enqueue again so the
                    # enqueue→train→remove pattern is preserved.
                    _state["post_session_queue"].enqueue(_entry)
                    enqueue_post_session_train(
                        conversation_id=_entry["session_id"],
                        transcript=_entry["transcript"],
                        speaker_id=_entry["speaker_id"],
                        speaker_name=_entry.get("speaker_name"),
                        loop=_state["consolidation_loop"],
                        background_trainer=_state["background_trainer"],
                        config=config,
                        state=_state,
                    )
    else:
        _state["post_session_queue"] = None

    logger.info("ParaMem server ready — mode: %s, model: %s", _state["mode"], config.model_name)

    yield

    # Shutdown — flush deferred speaker profile writes
    store = _state.get("speaker_store")
    if store:
        store.flush()

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
        tracker = _state.get("language_tracker")
        if tracker is not None:
            tracker.record(detected_language, detected_language_prob)

    # Speaker resolution: embedding → session history → anonymous.
    # Never let speaker ID failure kill the request — proceed as anonymous.
    try:
        speaker_id, speaker = _resolve_speaker(request, buffer, _state.get("speaker_store"))
    except Exception:
        logger.exception("Speaker resolution failed — proceeding as anonymous")
        speaker_id, speaker = None, None
    follow_up = None
    store = _state.get("speaker_store")

    # Deferred enrollment: unknown voice → group by embedding, prompt on first
    # encounter for each group, then re-prompt after per-group cooldown.
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
                group = {
                    "embeddings": [request.speaker_embedding],
                    "conversations": {conv_id},
                    "first_seen": now,
                    "last_prompted": None,
                    "last_extract_turn_count": 0,
                }
                _state["unknown_speakers"][unknown_group_id] = group
                _state["pending_enrollments"].add(conv_id)
                logger.info("Unknown speaker — new group %s", unknown_group_id)

            # Per-group enrollment prompt: fire on first encounter, re-prompt
            # after reprompt_interval seconds of the same unresolved group.
            reprompt_interval = _state["config"].speaker.enrollment_reprompt_interval
            last_prompted = group.get("last_prompted")
            if last_prompted is None or (now - last_prompted).total_seconds() >= reprompt_interval:
                follow_up = _state["config"].speaker.enrollment_prompt
                group["last_prompted"] = now
                logger.info(
                    "Enrollment prompt sent for group %s (interval %ds)",
                    unknown_group_id,
                    reprompt_interval,
                )
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

    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
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
                sota_agent=_state.get("sota_agent"),
                ha_client=_state.get("ha_client"),
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

    # Post-conversation training: after each assistant response, enqueue a
    # background job to extract and train onto the current interim adapter.
    # Only fires when:
    #   - a speaker is identified (speaker_id is required for key ownership)
    #   - post_session_train_enabled is True in the consolidation config
    #   - the consolidation loop exists (initialised at startup or by first
    #     consolidation run)
    # The job runs in a daemon thread — it never blocks the response path.
    # model handle is updated inside enqueue_post_session_train after training.
    if (
        speaker_id is not None
        and _state.get("consolidation_loop") is not None
        and _state["config"].consolidation.post_session_train_enabled
    ):
        transcript_turns = buffer.get_session_turns(request.conversation_id)
        if transcript_turns:
            transcript_text = "\n".join(
                f"{t.get('role', 'user')}: {t.get('text', '')}" for t in transcript_turns
            )
            # Persist the job to the queue BEFORE submitting to the trainer.
            # If the server crashes between here and the training completing,
            # the entry will be replayed on the next startup.
            _psq = _state.get("post_session_queue")
            if _psq is not None:
                _psq.enqueue(
                    {
                        "session_id": request.conversation_id,
                        "transcript": transcript_text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker,
                    }
                )
            enqueue_post_session_train(
                conversation_id=request.conversation_id,
                transcript=transcript_text,
                speaker_id=speaker_id,
                speaker_name=speaker,
                loop=_state["consolidation_loop"],
                background_trainer=_state.get("background_trainer"),
                config=_state["config"],
                state=_state,
                post_session_queue=_psq,
            )

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

    # Adapter inventory: enumerate configured kinds + interim capacity. Main
    # adapters contribute 1 each when enabled in yaml; interim contributes
    # max_interim_count (the capacity ceiling enforced by the VRAM validator).
    adapter_config_counts: dict[str, int] = {}
    for kind, cfg in (
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ):
        if cfg.enabled:
            adapter_config_counts[kind] = 1
    if config.consolidation.max_interim_count > 0:
        adapter_config_counts["interim"] = config.consolidation.max_interim_count

    # Currently active adapter on the live PeftModel. PEFT only keeps one
    # adapter active at a time (set_adapter / switch_adapter). None when the
    # model hasn't loaded (cloud-only) or no adapters exist yet.
    active_adapter: str | None = None
    if model is not None and hasattr(model, "active_adapter"):
        raw_active = model.active_adapter
        if isinstance(raw_active, list):
            active_adapter = raw_active[0] if raw_active else None
        else:
            active_adapter = raw_active

    keys_count = 0
    if config.registry_path.exists():
        with open(config.registry_path) as f:
            registry = json.load(f)
        keys_count = len(registry)

    # Session buffer summary (pending counts, orphan attribution, age)
    buf = _state.get("session_buffer")
    summary = (
        buf.get_summary()
        if buf
        else {"total": 0, "orphaned": 0, "oldest_age_seconds": None, "per_speaker": {}}
    )

    # Per-speaker profile snapshot enriched with pending-session counts
    store = _state.get("speaker_store")
    speaker_rows: list[dict] = []
    if store is not None:
        per_speaker = summary["per_speaker"]
        for prof in store.list_profiles():
            prof["pending"] = per_speaker.get(prof["id"], 0)
            speaker_rows.append(prof)

    # Next scheduled run — sourced from the systemd user timer (wall-clock,
    # survives server restart). See paramem/server/systemd_timer.py.
    # Cached for 5s because /status is polled frequently (HA, pstatus) and
    # `systemctl show` forks a subprocess on every call.
    from paramem.server import systemd_timer

    timer_state = systemd_timer.cached_timer_state(max_age_seconds=5)
    next_run_seconds: int | None = None
    scheduler_active = bool(timer_state.get("active", False))
    next_us = timer_state.get("next_elapse_us") or ""
    # systemd uses UINT64_MAX as the "no next elapse" sentinel. Treat any
    # timestamp > 100 years from now (1e11 seconds) as "not scheduled".
    if next_us.isdigit():
        next_epoch = int(next_us) / 1_000_000
        if next_epoch - time.time() < 3.15e9:  # < ~100 years ahead
            next_run_seconds = max(0, int(next_epoch - time.time()))

    # Next interim bucket boundary: post_session_train floors its stamp to
    # the refresh_cadence boundary measured from midnight, so the next
    # boundary is fully deterministic from the clock. None when cadence is
    # disabled (manual-only mode).
    from paramem.server.interim_adapter import compute_schedule_period_seconds

    next_interim_seconds: int | None = None
    _refresh_seconds = compute_schedule_period_seconds(config.consolidation.refresh_cadence)
    if _refresh_seconds and _refresh_seconds > 0:
        _now = datetime.now()
        _midnight = _now.replace(hour=0, minute=0, second=0, microsecond=0)
        _since_mid = int((_now - _midnight).total_seconds())
        _next_boundary = ((_since_mid // _refresh_seconds) + 1) * _refresh_seconds
        next_interim_seconds = max(0, _next_boundary - _since_mid)

    # Background trainer
    bt = _state.get("background_trainer")
    bg_active = bool(bt and getattr(bt, "is_training", False))
    bg_adapter = getattr(bt, "current_adapter_name", None) if bg_active else None

    # Thermal-throttle / quiet-hours snapshot. Read from the loaded config so the
    # block is present even before the BackgroundTrainer has been constructed.
    # ``currently_throttling`` reflects the policy gate only; the actual throttle
    # additionally requires ``training_temp_limit > 0`` and temp above limit —
    # surfaced here is "would the policy allow throttling right now".
    from paramem.server.background_trainer import is_thermal_policy_active

    thermal_policy = {
        "mode": config.consolidation.quiet_hours_mode,
        "start": config.consolidation.quiet_hours_start,
        "end": config.consolidation.quiet_hours_end,
        "temp_limit": config.consolidation.training_temp_limit,
        "currently_throttling": is_thermal_policy_active(
            config.consolidation.quiet_hours_mode,
            config.consolidation.quiet_hours_start,
            config.consolidation.quiet_hours_end,
        ),
    }

    # Adapter health — stored in the KeyRegistry JSON (distinct from the
    # SimHash registry at config.registry_path).  Surfaced to pstatus so a
    # degenerated adapter is visible without grepping logs.
    adapter_health: dict = {}
    _key_reg_path = config.adapter_dir / "indexed_key_registry.json"
    if _key_reg_path.exists():
        try:
            with open(_key_reg_path) as f:
                _key_reg_json = json.load(f)
            adapter_health = _key_reg_json.get("adapter_health", {}) or {}
        except (OSError, json.JSONDecodeError):
            adapter_health = {}

    # TTS inventory: which languages are loaded and on which device. When
    # voices span devices (one on CUDA, one on CPU) we report "mixed" so the
    # fallback path is visible in pstatus without dumping per-voice rows.
    tts_manager = _state.get("tts_manager")
    tts_loaded = bool(tts_manager and tts_manager.is_loaded)
    tts_languages: list[str] = tts_manager.available_languages if tts_loaded else []
    tts_device: str | None = None
    if tts_loaded:
        _tts_devices = set(tts_manager.engine_devices.values())
        if len(_tts_devices) == 1:
            tts_device = next(iter(_tts_devices))
        elif _tts_devices:
            tts_device = "mixed"

    stt = _state.get("stt")
    stt_loaded = stt is not None and stt.is_loaded
    # WhisperSTT keeps `self.device` as the RESOLVED device string (cuda/cpu)
    # by the time load() returns True — "auto" is reassigned before load.
    stt_device = stt.device if stt_loaded else None
    # Only one STT backend family is supported today (faster-whisper).
    stt_engine = "whisper" if stt_loaded else None

    # TTS engine family: derive from the class name of each loaded engine
    # (piper / mms_tts). "piper+mms" when voices span both backends.
    tts_engine: str | None = None
    if tts_loaded:
        _kinds: set[str] = set()
        for _eng in tts_manager._engines.values():
            _cls = type(_eng).__name__.lower()
            if "piper" in _cls:
                _kinds.add("piper")
            elif "mms" in _cls:
                _kinds.add("mms_tts")
        if len(_kinds) == 1:
            tts_engine = next(iter(_kinds))
        elif _kinds:
            tts_engine = "piper+mms"

    # Live device of the loaded LLM. Resolved from the first parameter's
    # device so we reflect actual placement, not the config intent (which
    # can diverge in cloud-only or CPU-fallback cases).
    model_device: str | None = None
    if model is not None:
        try:
            model_device = next(model.parameters()).device.type
        except (StopIteration, AttributeError):
            model_device = None

    # HF model identifier from the registry. Safe even in cloud-only mode —
    # model_config resolves off the registry and has no GPU dependency.
    model_id: str | None = None
    try:
        model_id = config.model_config.model_id
    except (KeyError, ValueError):
        model_id = None

    # Episodic adapter rank surfaces the primary knob for indexed-key recall.
    episodic_rank = config.adapters.episodic.rank if config.adapters.episodic.enabled else None

    # Per-kind adapter spec. Only include enabled kinds so pstatus doesn't
    # display disabled rows. target_kind compresses target_modules into a
    # category label — "attn+mlp" means MLP layers are in the set, else
    # "attn". A caller interested in the exact list can hit the yaml.
    def _target_kind(target_modules: list[str]) -> str:
        for t in target_modules or []:
            tl = t.lower()
            if "mlp" in tl or "gate" in tl or "up_proj" in tl or "down_proj" in tl:
                return "attn+mlp"
        return "attn"

    adapter_specs: dict[str, dict] = {}
    for _kind, _cfg in (
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ):
        if not _cfg.enabled:
            continue
        adapter_specs[_kind] = {
            "rank": _cfg.rank,
            "alpha": _cfg.alpha,
            "learning_rate": _cfg.learning_rate,
            "target_kind": _target_kind(_cfg.target_modules),
        }

    # Speaker-embedding backend. Only populate when the pyannote model is
    # actually loaded — disabled / failed-load paths leave the fields None
    # so pstatus can skip the row entirely.
    speaker_embedding_backend: str | None = None
    speaker_embedding_model: str | None = None
    speaker_embedding_device: str | None = None
    try:
        from paramem.server import speaker_embedding as _spk_emb

        if _spk_emb.is_loaded():
            speaker_embedding_backend = _spk_emb.EMBEDDING_BACKEND
            speaker_embedding_model = _spk_emb.EMBEDDING_MODEL_NAME
            speaker_embedding_device = _spk_emb.EMBEDDING_DEVICE
    except ImportError:
        pass

    return StatusResponse(
        model=config.model_name,
        model_id=model_id,
        model_device=model_device,
        episodic_rank=episodic_rank,
        adapter_specs=adapter_specs,
        speaker_embedding_backend=speaker_embedding_backend,
        speaker_embedding_model=speaker_embedding_model,
        speaker_embedding_device=speaker_embedding_device,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        mode=_state["mode"],
        cloud_only_reason=_state.get("cloud_only_reason"),
        adapter_loaded=adapter_loaded,
        adapter_config=adapter_config_counts,
        active_adapter=active_adapter,
        keys_count=keys_count,
        pending_sessions=summary["total"],
        consolidating=_state["consolidating"],
        last_consolidation=_state["last_consolidation"],
        speaker_profiles=store.profile_count if store else 0,
        stt_loaded=stt_loaded,
        stt_model=stt.model_name if stt_loaded else None,
        stt_device=stt_device,
        tts_loaded=tts_loaded,
        tts_languages=tts_languages,
        tts_device=tts_device,
        refresh_cadence=config.consolidation.refresh_cadence,
        consolidation_period=config.consolidation.consolidation_period_string,
        max_interim_count=config.consolidation.max_interim_count,
        mode_config=config.consolidation.mode,
        next_run_seconds=next_run_seconds,
        next_interim_seconds=next_interim_seconds,
        orphaned_pending=summary["orphaned"],
        oldest_pending_seconds=summary["oldest_age_seconds"],
        speakers=speaker_rows,
        bg_trainer_active=bg_active,
        bg_trainer_adapter=bg_adapter,
        thermal_policy=thermal_policy,
        last_consolidation_result=_state.get("last_consolidation_result"),
        pending_enrollments=len(_state.get("pending_enrollments") or []),
        scheduler_started=scheduler_active,
        adapter_health=adapter_health,
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


@app.post("/debug/assign-orphans")
async def debug_assign_orphans(speaker_id: str | None = None):
    """Test-only: attribute all orphan sessions to a single speaker.

    Requires debug mode. Uses the given speaker_id, or the first
    enrolled profile if omitted. Mutates in-memory turns and, in
    debug persistence mode, rewrites the jsonl files on disk.
    """
    config = _state["config"]
    if not getattr(config, "debug", False):
        return {"status": "forbidden_not_debug"}
    store = _state.get("speaker_store")
    buffer = _state.get("session_buffer")
    if store is None or buffer is None:
        return {"status": "not_ready"}
    profiles = store.list_profiles()
    if not profiles:
        return {"status": "no_speakers_enrolled"}
    target = next((p for p in profiles if p["id"] == speaker_id), profiles[0])
    sid, sname = target["id"], target["name"]
    claimed = 0
    for conv_id, turns in buffer._turns.items():
        if any(t.get("speaker_id") for t in turns):
            continue
        for turn in turns:
            turn["speaker"] = sname
            turn["speaker_id"] = sid
        claimed += 1
        if buffer.debug:
            path = buffer.session_dir / f"{conv_id}.jsonl"
            if path.exists():
                with open(path, "w") as f:
                    for turn in turns:
                        f.write(json.dumps(turn) + "\n")
    logger.info("debug/assign-orphans: %d sessions → speaker %s (%s)", claimed, sname, sid)
    return {"status": "ok", "claimed": claimed, "speaker": sname, "speaker_id": sid}


@app.post("/scheduled-tick", response_model=ConsolidateResponse)
async def scheduled_tick():
    """Systemd user-timer entrypoint (paramem-consolidate.timer).

    Runs the cooperative extract + background-train path. If the GPU is
    unavailable (cloud-only or bg training active), returns a 'deferred'
    status — the timer will fire again on its next wall-clock tick.
    """
    _state["scheduler_last_tick_epoch"] = time.time()
    status = _maybe_trigger_scheduled_consolidation()
    _state["scheduler_last_tick_status"] = status
    return ConsolidateResponse(status=status)


@app.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate():
    """Trigger consolidation manually."""
    if _state["mode"] == "cloud-only":
        return ConsolidateResponse(status="rejected_cloud_only")
    if _state["consolidating"]:
        return ConsolidateResponse(status="already_running")

    bg_trainer = _state.get("background_trainer")
    if bg_trainer is not None and bg_trainer.is_training:
        return ConsolidateResponse(status="training_active")

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
    completion. Holds the GPU lock for the entire consolidation to
    prevent concurrent CUDA access from STT/TTS/inference.
    """
    from paramem.server.gpu_lock import gpu_lock_sync

    with gpu_lock_sync():
        _run_consolidation_inner()


def _run_consolidation_inner():
    """Consolidation logic — must be called with GPU lock held."""
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
    _state["last_consolidation_result"] = {k: v for k, v in result.items() if k != "loop"}
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


def _maybe_trigger_scheduled_consolidation() -> str:
    """Gate + dispatch the scheduled extract + BG-train run.

    Returns a short status string. GPU-busy states return 'deferred_*' so the
    caller can distinguish a missed-but-rescheduled tick from a true no-op.
    The systemd timer will fire again on its next wall-clock tick and retry.
    """
    if _state["consolidating"]:
        logger.info("Scheduler tick: consolidation already running — deferred")
        return "deferred_already_running"
    if _state["mode"] != "local":
        logger.info(
            "Scheduler tick: mode=%s (reason=%s) — deferred, will retry on next tick",
            _state["mode"],
            _state.get("cloud_only_reason"),
        )
        return "deferred_cloud_only"
    bg = _state.get("background_trainer")
    if bg is not None and bg.is_training:
        logger.info("Scheduler tick: background training active — deferred")
        return "deferred_bg_training"

    buffer = _state["session_buffer"]

    # Retroactive voice-match claim: scan orphan sessions against every
    # enrolled speaker. Attributes sessions whose embeddings match an
    # existing profile at high confidence. Cheap — centroids are cached.
    _retro_claim_orphan_sessions()

    pending = buffer.get_pending()
    if not pending:
        logger.info("Scheduler tick: no pending sessions — noop")
        return "noop_no_pending"
    if not any(s.get("speaker_id") for s in pending):
        logger.info("Scheduler tick: no sessions with speaker_id — noop")
        return "noop_no_speaker"

    logger.info("Scheduler tick: %d pending sessions, starting extract + train", len(pending))
    _state["consolidating"] = True

    event_loop = asyncio.get_running_loop()
    future = event_loop.run_in_executor(None, _extract_and_start_training)
    future.add_done_callback(_scheduled_extract_done_callback)
    return "started"


def _retro_claim_orphan_sessions() -> int:
    """Attribute orphaned pending sessions to existing speaker profiles via voice match.

    Runs before each scheduled consolidation tick. For each enrolled speaker,
    invokes `SessionBuffer.claim_sessions_for_speaker` which scans orphan
    sessions for user turns whose stored embeddings match the speaker at
    high confidence. Idempotent — already-claimed sessions are skipped.

    Runs on the asyncio event loop thread (same thread as `_enrollment_idle_loop`),
    so there is no lock contention over `SessionBuffer._turns` with the
    enrollment path. `_extract_and_start_training` runs in an executor but is
    dispatched only after this function returns.

    A corrupt profile or session must not take down the entire tick —
    each speaker is wrapped independently.

    Returns the total number of sessions claimed across all speakers.
    """
    buffer = _state.get("session_buffer")
    store = _state.get("speaker_store")
    if buffer is None or store is None:
        return 0

    total = 0
    for profile in store.list_profiles():
        try:
            claimed = buffer.claim_sessions_for_speaker(profile["id"], profile["name"], store)
        except Exception:
            logger.exception(
                "Retro-claim failed for speaker %s (%s) — skipping",
                profile.get("name", "?"),
                profile.get("id", "?"),
            )
            continue
        total += claimed
    if total > 0:
        logger.info("Retro-claim: attributed %d orphan sessions to known speakers", total)
    else:
        logger.debug("Retro-claim: no new orphan sessions matched known speakers")
    return total


def _scheduled_extract_done_callback(future):
    """Clear the consolidating flag only if the extraction phase failed.

    On success, _extract_and_start_training has handed off to the background
    trainer and the flag will be cleared by _finalize_background_training.
    """
    exc = future.exception()
    if exc:
        logger.exception("Scheduled extraction failed: %s", exc)
        _state["consolidating"] = False


def _extract_and_start_training():
    """Extract sessions, prepare training data, start background trainer.

    Runs in executor thread. Extraction holds the GPU lock (LLM inference).
    Training runs in a background thread and acquires/releases the GPU lock
    per training step via the BackgroundTrainer pause/resume mechanism.
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

    # --- Phase 1: Extract all sessions (holds GPU lock) ---
    from paramem.server.gpu_lock import gpu_lock_sync

    pending = session_buffer.get_pending()
    all_episodic_qa = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []

    with gpu_lock_sync():
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

            speaker_name = None
            if _state.get("speaker_store") is not None:
                try:
                    speaker_name = _state["speaker_store"].get_name(session_speaker_id)
                except Exception as e:
                    logger.warning("speaker_store.get_name(%s) failed: %s", session_speaker_id, e)

            episodic_qa, procedural_rels = loop.extract_session(
                transcript,
                session_id,
                speaker_id=session_speaker_id,
                speaker_name=speaker_name,
                ha_context=ha_context,
                stt_correction=config.consolidation.extraction_stt_correction,
                ha_validation=config.consolidation.extraction_ha_validation,
                noise_filter=config.consolidation.extraction_noise_filter,
                noise_filter_model=config.consolidation.extraction_noise_filter_model,
                noise_filter_endpoint=config.consolidation.extraction_noise_filter_endpoint or None,
                ner_check=config.consolidation.extraction_ner_check,
                ner_model=config.consolidation.extraction_ner_model,
                plausibility_judge=config.consolidation.extraction_plausibility_judge,
                plausibility_stage=config.consolidation.extraction_plausibility_stage,
                verify_anonymization=config.consolidation.extraction_verify_anonymization,
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
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        _state["last_consolidation_result"] = {
            "status": "no_facts",
            "sessions": len(session_ids),
            "episodic_qa": 0,
            "procedural_rels": 0,
        }
        _state["consolidating"] = False
        return

    # --- Simulate mode: save results, skip training ---
    if config.consolidation.mode == "simulate":
        if config.debug:
            _save_simulation_results(all_episodic_qa, all_procedural_rels, loop, config)
        session_buffer.mark_consolidated(session_ids)
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        _state["last_consolidation_result"] = {
            "status": "simulated",
            "sessions": len(session_ids),
            "episodic_qa": len(all_episodic_qa),
            "procedural_rels": len(all_procedural_rels),
        }
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
        temp_limit=config.consolidation.training_temp_limit,
        temp_check_interval=config.consolidation.training_temp_check_interval,
        quiet_hours_mode=config.consolidation.quiet_hours_mode,
        quiet_hours_start=config.consolidation.quiet_hours_start,
        quiet_hours_end=config.consolidation.quiet_hours_end,
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
    total_keys = (
        len(loop.episodic_simhash) + len(loop.semantic_simhash) + len(loop.procedural_simhash)
    )
    _state["last_consolidation_result"] = {
        "status": "trained",
        "total_keys": total_keys,
        "jobs": [job[0] for job in (jobs_data or [])],
    }
    _state["consolidating"] = False
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

        from paramem.server.gpu_lock import gpu_lock

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

            # Skip re-extraction when no new turns arrived since last attempt.
            if len(all_turns) == group.get("last_extract_turn_count", 0):
                continue
            group["last_extract_turn_count"] = len(all_turns)

            async with gpu_lock():
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
                    _state["pending_enrollments"] -= group["conversations"]
                    del _state["unknown_speakers"][group_id]
                else:
                    logger.info(
                        "Deferred enrollment skipped for group %s (voice already enrolled)",
                        group_id,
                    )
                    _state["pending_enrollments"] -= group["conversations"]
                    del _state["unknown_speakers"][group_id]
            else:
                # Retain group: speaker may introduce themselves in a later
                # utterance, or retro-claim may match them to an existing
                # profile, or an admin may attach manually. Re-prompt fires
                # via per-group cooldown on the next utterance after interval.
                logger.info(
                    "No self-introduction found in group %s — retaining for re-prompt",
                    group_id,
                )


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

    # Clear stale systemd env var so it can't resurface on next restart.
    # tresume sets PARAMEM_EXTRA_ARGS=--defer-model; tpause normally clears
    # it, but if a training run was paused without tpause (e.g. killed),
    # the var persists and every restart silently starts cloud-only. When
    # this process starts WITHOUT --defer-model, treat it as authoritative
    # and clear the var so the intent is consistent with future restarts.
    if not args.defer_model and not args.cloud_only:
        import subprocess

        try:
            subprocess.run(
                ["systemctl", "--user", "unset-environment", "PARAMEM_EXTRA_ARGS"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

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
