"""ParaMem server — REST API wrapping the parametric memory pipeline.

Usage:
    python -m paramem.server.app --config configs/server.yaml
"""

import argparse
import asyncio
import logging
import os
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
    "consolidating": False,
    "last_consolidation": None,
    "scheduler_task": None,
}


# --- Request/Response schemas ---


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    text: str
    escalated: bool = False


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
    _state["session_buffer"] = SessionBuffer(config.session_dir)
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
        graph_path=config.graph_path,
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
    """Handle a conversation turn. Adapter is always active."""
    loop = asyncio.get_event_loop()
    result: ChatResult = await loop.run_in_executor(
        None,
        lambda: handle_chat(
            text=request.text,
            conversation_id=request.conversation_id,
            history=request.history,
            model=_state["model"],
            tokenizer=_state["tokenizer"],
            config=_state["config"],
            router=_state["router"],
        ),
    )

    # Buffer the conversation turn
    buffer = _state["session_buffer"]
    buffer.append(request.conversation_id, "user", request.text)
    buffer.append(request.conversation_id, "assistant", result.text)

    return ChatResponse(text=result.text, escalated=result.escalated)


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
        import json

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

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_consolidation_sync)

    return ConsolidateResponse(status="started")


# --- Internal ---


def _run_consolidation_sync():
    """Run consolidation in a background thread."""
    _state["consolidating"] = True
    try:
        result = run_consolidation(
            model=_state["model"],
            tokenizer=_state["tokenizer"],
            config=_state["config"],
            session_buffer=_state["session_buffer"],
        )
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        _state["router"].reload()
        logger.info("Consolidation result: %s", result)
    except Exception:
        logger.exception("Consolidation failed")
    finally:
        _state["consolidating"] = False


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
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, _run_consolidation_sync)
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
