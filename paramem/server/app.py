"""ParaMem server — REST API wrapping the parametric memory pipeline.

Usage:
    python -m paramem.server.app --config configs/server.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from paramem.models.loader import load_adapter, load_base_model, unload_model
from paramem.server.config import load_server_config
from paramem.server.consolidation import run_consolidation
from paramem.server.inference import ChatResult, handle_chat
from paramem.server.router import QueryRouter
from paramem.server.session_buffer import SessionBuffer

logger = logging.getLogger(__name__)

# Global state — single model, single adapter, single server
_state = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "session_buffer": None,
    "router": None,
    "consolidation_loop": None,
    "consolidating": False,
    "last_consolidation": None,
    "scheduler_task": None,
}


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

    # Start consolidation scheduler if configured
    if config.consolidation.schedule:
        _state["scheduler_task"] = asyncio.create_task(
            _consolidation_scheduler(config.consolidation.schedule)
        )
        logger.info("Consolidation scheduled at: %s", config.consolidation.schedule)

    logger.info("ParaMem server ready — model: %s", config.model_name)

    yield

    # Shutdown
    if _state["scheduler_task"]:
        _state["scheduler_task"].cancel()
    if _state["model"]:
        unload_model(_state["model"], _state["tokenizer"])
        logger.info("Model unloaded")


app = FastAPI(title="ParaMem", version="0.1.0", lifespan=lifespan)


# --- Endpoints ---


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a conversation turn with speaker identification."""
    buffer = _state["session_buffer"]

    # If the request provides a speaker (e.g. from HA user context), store it
    if request.speaker:
        buffer.set_speaker(request.conversation_id, request.speaker)

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

    # Speaker identified — normal inference with entity routing
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
        ),
    )

    buffer.append(request.conversation_id, "user", request.text)
    buffer.append(request.conversation_id, "assistant", result.text)

    return ChatResponse(text=result.text, escalated=result.escalated, speaker=speaker)


def _extract_speaker_name(text: str) -> str | None:
    """Extract a speaker name from a self-introduction response.

    Handles common patterns: "I'm Tobias", "My name is Tobias",
    "Tobias", "It's Tobias", "This is Tobias", "Call me Tobias".
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
        adapter_loaded=adapter_loaded,
        keys_count=keys_count,
        pending_sessions=_state["session_buffer"].pending_count if _state["session_buffer"] else 0,
        consolidating=_state["consolidating"],
        last_consolidation=_state["last_consolidation"],
    )


@app.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate():
    """Trigger consolidation manually."""
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
            if not triggered_today and not _state["consolidating"]:
                logger.info("Scheduled consolidation triggered at %s", schedule)
                _state["consolidating"] = True
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, _run_consolidation_sync)
                future.add_done_callback(_consolidation_done_callback)
                triggered_today = True
        else:
            triggered_today = False

        await asyncio.sleep(30)


# --- Entry point ---


def main():
    parser = argparse.ArgumentParser(description="ParaMem Server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/server.yaml",
        help="Path to server config YAML",
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

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
