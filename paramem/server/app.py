"""ParaMem server — REST API wrapping the parametric memory pipeline.

Usage:
    python -m paramem.server.app --config configs/server.yaml

GPU lifecycle:
    SIGUSR1 → release GPU (switch to cloud-only mode)
    Auto-reclaim timer reloads model when GPU is free
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import re
import signal
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from paramem.models.loader import load_adapter, load_base_model, unload_model
from paramem.server.cloud import get_cloud_agent
from paramem.server.config import load_server_config
from paramem.server.consolidation import run_consolidation
from paramem.server.inference import ChatResult, handle_chat
from paramem.server.router import QueryRouter
from paramem.server.session_buffer import SessionBuffer
from paramem.server.tools.ha_client import HAClient
from paramem.server.tools.registry import ToolRegistry
from paramem.utils.notify import SERVER_CLOUD_ONLY, SERVER_RECLAIMED, notify_server

logger = logging.getLogger(__name__)

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
    "reclaim_task": None,
    "mode": "local",  # "local" or "cloud-only"
}

# Lock to prevent concurrent model access during mode transitions
_mode_lock = asyncio.Lock()


# --- Request/Response schemas ---


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"
    speaker: str | None = None
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    text: str
    escalated: bool = False
    speaker: str | None = None


class StatusResponse(BaseModel):
    model: str
    mode: str  # "local" or "cloud-only"
    adapter_loaded: bool
    keys_count: int
    pending_sessions: int
    consolidating: bool
    last_consolidation: str | None


class ConsolidateResponse(BaseModel):
    status: str


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    config = _state["config"]

    cloud_only = _state.get("cloud_only_startup", False)

    # Auto-detect GPU conflict: if another process holds the GPU, start cloud-only
    if not cloud_only and _gpu_occupied():
        logger.warning(
            "GPU is occupied by another process — starting in cloud-only mode. "
            "Will auto-reclaim when GPU is free."
        )
        notify_server(SERVER_CLOUD_ONLY)
        cloud_only = True

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
    )
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
    )

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

    # Initialize HA client and tool registry if configured
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
        else:
            logger.warning("HA client: configured but unreachable at %s", tools_config.ha.url)
        _state["ha_client"] = ha_client

        # Load tool definitions
        registry = ToolRegistry()
        if tools_config.ha.auto_discover:
            ha_services = ha_client.get_services()
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

    # Start consolidation scheduler if configured
    if config.consolidation.schedule:
        _state["scheduler_task"] = asyncio.create_task(
            _consolidation_scheduler(config.consolidation.schedule)
        )
        logger.info("Consolidation scheduled at: %s", config.consolidation.schedule)

    # Register SIGUSR1 handler for GPU release
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    logger.info("SIGUSR1 handler registered (send to release GPU)")

    # Start auto-reclaim timer
    reclaim_interval = getattr(config.server, "reclaim_interval_minutes", 10)
    _state["reclaim_task"] = asyncio.create_task(_auto_reclaim_loop(reclaim_interval))

    _state["mode"] = "cloud-only" if cloud_only else "local"
    logger.info("ParaMem server ready — mode: %s, model: %s", _state["mode"], config.model_name)

    yield

    # Shutdown
    if _state["scheduler_task"]:
        _state["scheduler_task"].cancel()
    if _state["reclaim_task"]:
        _state["reclaim_task"].cancel()
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
    buffer = _state["session_buffer"]

    # Ignore application-level user identity (e.g. HA user context).
    # Speaker must self-identify in the conversation via the greeting flow.

    # Speaker identification flow
    state = buffer.get_session_state(request.conversation_id)
    speaker = buffer.get_speaker(request.conversation_id)

    if state == "new":
        buffer.set_state(request.conversation_id, "greeting_sent")
        greeting = "Hi, I'm your home assistant. Who are you?"
        return ChatResponse(text=greeting, speaker=speaker)

    if state == "greeting_sent":
        extracted_name = _extract_speaker_name(request.text)
        if extracted_name:
            buffer.set_speaker(request.conversation_id, extracted_name)
            speaker = extracted_name
            welcome = f"Nice to meet you, {extracted_name}. I'll memorize everything we discuss."
            buffer.append(request.conversation_id, "user", request.text)
            buffer.append(request.conversation_id, "assistant", welcome)
            return ChatResponse(text=welcome, speaker=speaker)
        else:
            # Could not extract name — ask again
            return ChatResponse(
                text="I didn't catch your name. Could you tell me who you are?",
                speaker=speaker,
            )

    # Cloud-only mode — forward to HA conversation agent
    if _state["mode"] == "cloud-only":
        ha_client = _state.get("ha_client")
        if ha_client is None:
            return ChatResponse(
                text="I'm running in limited mode right now. Please try again later.",
                speaker=speaker,
            )
        response_text = ha_client.conversation_process(
            request.text, agent_id=_state["config"].ha_agent_id
        )
        if response_text is None:
            response_text = "I couldn't get an answer right now. Please try again."
        buffer.append(request.conversation_id, "user", request.text)
        buffer.append(request.conversation_id, "assistant", response_text)
        return ChatResponse(text=response_text, escalated=True, speaker=speaker)

    # Local mode — normal inference with entity routing
    async with _mode_lock:
        loop = asyncio.get_running_loop()
        result: ChatResult = await loop.run_in_executor(
            None,
            lambda: handle_chat(
                text=request.text,
                conversation_id=request.conversation_id,
                speaker=speaker,
                history=request.history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=_state["config"],
                router=_state["router"],
                cloud_agent=_state.get("cloud_agent"),
                ha_client=_state.get("ha_client"),
                tool_registry=_state.get("tool_registry"),
            ),
        )

    buffer.append(request.conversation_id, "user", request.text)
    buffer.append(request.conversation_id, "assistant", result.text)

    return ChatResponse(text=result.text, escalated=result.escalated, speaker=speaker)


def _extract_speaker_name(text: str) -> str | None:
    """Extract a speaker name from a self-introduction response.

    Handles common patterns: "I'm Alex", "My name is Alex",
    "Alex", "It's Alex", "This is Alex", "Call me Alex".
    """

    text = text.strip().rstrip(".")

    patterns = [
        r"(?:I'm|I am|my name is|it's|this is|call me|they call me)\s+(\w+)",
        r"^(\w+)$",  # bare name
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1)
            # Filter out non-name words
            if name.lower() not in ("hi", "hello", "hey", "yes", "no", "the", "a"):
                return name.capitalize()
    return None


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
        adapter_loaded=adapter_loaded,
        keys_count=keys_count,
        pending_sessions=_state["session_buffer"].pending_count if _state["session_buffer"] else 0,
        consolidating=_state["consolidating"],
        last_consolidation=_state["last_consolidation"],
    )


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


# --- Internal ---


def _run_consolidation_sync():
    """Run consolidation in a background thread.

    The consolidating flag is set by the caller before submitting to
    the executor, and cleared by _consolidation_done_callback after
    completion. This eliminates the race window.
    """
    result = run_consolidation(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        config=_state["config"],
        session_buffer=_state["session_buffer"],
        loop=_state["consolidation_loop"],
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


# --- GPU lifecycle ---


def _release_gpu():
    """Unload model from VRAM, switch to cloud-only mode.

    Called from SIGUSR1 handler. Blocks until model is fully unloaded.
    """
    if _state["mode"] == "cloud-only":
        logger.info("Already in cloud-only mode")
        return

    if _state["consolidating"]:
        logger.warning("Cannot release GPU — consolidation in progress")
        return

    logger.info("Releasing GPU — switching to cloud-only mode")

    # Unload model and free VRAM
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    _state["model"] = None
    _state["tokenizer"] = None
    _state["consolidation_loop"] = None

    if model is not None:
        unload_model(model, tokenizer)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    _state["mode"] = "cloud-only"

    # Suspend consolidation scheduler
    if _state["scheduler_task"]:
        _state["scheduler_task"].cancel()
        _state["scheduler_task"] = None
        logger.info("Consolidation scheduler suspended")

    notify_server(SERVER_CLOUD_ONLY)
    logger.info("GPU released — cloud-only mode active")


def _reclaim_gpu():
    """Reload model and adapters, switch back to local mode.

    Called by the auto-reclaim timer when the GPU is free.
    """
    if _state["mode"] == "local":
        logger.info("Already in local mode")
        return

    config = _state["config"]
    logger.info("Reclaiming GPU — loading model: %s", config.model_name)

    model, tokenizer = load_base_model(config.model_config)

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

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["router"].reload()

    # Restore consolidation scheduler — event_loop is passed from the async caller
    # because _reclaim_gpu runs in a thread pool via run_in_executor
    if config.consolidation.schedule and not _state["scheduler_task"]:
        event_loop = _state.get("_event_loop")
        if event_loop:
            event_loop.call_soon_threadsafe(
                lambda: _state.update(
                    scheduler_task=event_loop.create_task(
                        _consolidation_scheduler(config.consolidation.schedule)
                    )
                )
            )
            logger.info("Consolidation scheduler restored")

    _state["mode"] = "local"
    notify_server(SERVER_RECLAIMED)
    logger.info("GPU reclaimed — local mode active")


def _gpu_occupied() -> bool:
    """Check if another process is using the GPU at startup time.

    Used to prevent loading the model when an ML workload is running.
    """
    return _gpu_has_compute_processes()


def _gpu_has_compute_processes() -> bool:
    """Check if any non-server process is using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
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
        logger.warning("nvidia-smi unavailable (%s) — assuming GPU occupied", type(e).__name__)
        return True


async def _auto_reclaim_loop(interval_minutes: int = 10):
    """Periodically check if GPU is free and reclaim it.

    Only runs while in cloud-only mode. Checks every interval_minutes
    whether any GPU compute process is still running. If not, reloads
    the model. Handles the SIGKILL case where acquire_gpu's cleanup
    never fires.
    """
    interval_seconds = interval_minutes * 60
    while True:
        await asyncio.sleep(interval_seconds)
        if _state["mode"] != "cloud-only":
            continue
        if _gpu_has_compute_processes():
            logger.debug("Auto-reclaim: GPU still occupied, waiting")
            continue
        logger.info("Auto-reclaim: GPU free, reclaiming")
        async with _mode_lock:
            loop = asyncio.get_running_loop()
            _state["_event_loop"] = loop
            await loop.run_in_executor(None, _reclaim_gpu)


def _handle_sigusr1(signum, frame):
    """Signal handler for SIGUSR1 — schedule GPU release on the event loop."""
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(asyncio.ensure_future, _async_release_gpu())


async def _async_release_gpu():
    """Async wrapper for GPU release — acquires lock first."""
    async with _mode_lock:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _release_gpu)


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
        help="Start in cloud-only mode (skip model loading, no GPU)",
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

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
