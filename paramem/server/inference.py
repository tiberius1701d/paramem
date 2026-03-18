"""Chat inference — formats prompts and generates responses with the adapter active.

Two inference paths:
1. Standard: adapter active, facts in weights, model answers directly.
2. Temporal: detect time reference → filter registry by date → probe matching
   keys → feed recalled facts as context → generate answer.
"""

import json
import logging
from dataclasses import dataclass, field

from paramem.evaluation.recall import generate_answer
from paramem.models.loader import adapt_messages
from paramem.server.config import ServerConfig
from paramem.server.escalation import detect_escalation, escalate_to_cloud
from paramem.server.temporal import detect_temporal_query, filter_registry_by_date

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


@dataclass
class ChatResult:
    text: str
    escalated: bool = False
    temporal_keys: list[str] = field(default_factory=list)


def handle_chat(
    text: str,
    conversation_id: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
) -> ChatResult:
    """Process a chat message and generate a response.

    The adapter is always active — personal facts are in the weights.
    Temporal queries trigger registry-based key lookup and recall+reason.
    If the model emits [ESCALATE], the query is forwarded to the cloud.
    """
    model.gradient_checkpointing_disable()

    # Check for temporal reference
    date_range = detect_temporal_query(text)
    if date_range:
        return _handle_temporal_query(
            text, date_range, history, model, tokenizer, config
        )

    return _handle_standard_query(text, history, model, tokenizer, config)


def _handle_standard_query(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
) -> ChatResult:
    """Standard path — facts are in the weights, model answers directly."""
    messages = _build_messages(text, history, config.voice.system_prompt, tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate_answer(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        repetition_penalty=1.1,
    )

    should_escalate, forwarded_query = detect_escalation(response)

    if should_escalate and config.cloud.enabled:
        logger.info("Escalating to cloud: %s", forwarded_query[:100])
        cloud_response = escalate_to_cloud(forwarded_query, config.cloud)
        if cloud_response:
            return ChatResult(text=cloud_response, escalated=True)

    if should_escalate:
        response = response.replace("[ESCALATE]", "").strip()
        if not response:
            response = "I'm not sure about that."

    return ChatResult(text=response, escalated=False)


def _handle_temporal_query(
    text: str,
    date_range: tuple,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
) -> ChatResult:
    """Temporal path — probe registry keys from the date range, feed as context."""
    from paramem.training.indexed_memory import probe_key

    start_date, end_date = date_range
    logger.info("Temporal query detected: %s to %s", start_date, end_date)

    matching_keys = filter_registry_by_date(config.registry_path, start_date, end_date)
    if not matching_keys:
        logger.info("No keys found for date range %s to %s", start_date, end_date)
        return _handle_standard_query(text, history, model, tokenizer, config)

    logger.info("Found %d keys for date range, probing...", len(matching_keys))

    # Load registry for confidence verification
    registry = {}
    if config.registry_path.exists():
        with open(config.registry_path) as f:
            raw = json.load(f)
        # Extract simhash values for probe_key verification
        for key, meta in raw.items():
            if isinstance(meta, dict):
                registry[key] = meta.get("simhash", 0)

    # Probe matching keys to reconstruct facts
    recalled_facts = []
    for key in matching_keys:
        result = probe_key(model, tokenizer, key, registry=registry)
        if result and "failure_reason" not in result:
            recalled_facts.append(
                f"Q: {result.get('question', '')}\nA: {result.get('answer', '')}"
            )

    if not recalled_facts:
        logger.info("All probed keys failed, falling back to standard path")
        return _handle_standard_query(text, history, model, tokenizer, config)

    # Build prompt with recalled facts as context
    facts_context = "\n\n".join(recalled_facts)
    augmented_text = (
        f"Based on the following facts from that time period:\n\n"
        f"{facts_context}\n\n"
        f"Answer the question: {text}"
    )

    messages = _build_messages(augmented_text, history, config.voice.system_prompt, tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate_answer(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        repetition_penalty=1.1,
    )

    return ChatResult(text=response, temporal_keys=matching_keys)


def _build_messages(
    text: str,
    history: list[dict] | None,
    system_prompt: str,
    tokenizer,
) -> list[dict]:
    """Build chat messages from user text, history, and system prompt."""
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role", "user")
            content = turn.get("text", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": text})

    return adapt_messages(messages, tokenizer)
