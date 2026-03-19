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
    probed_keys: list[str] = field(default_factory=list)


def handle_chat(
    text: str,
    conversation_id: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    router=None,
) -> ChatResult:
    """Process a chat message and generate a response.

    Three inference paths, checked in order:
    1. Temporal: time reference detected → probe keys by date range
    2. Routed: entities found in query → probe targeted keys per adapter
    3. Standard: no entities or router → answer from active adapter weights

    The adapter is always active. Router is optional — without it,
    all queries go through the standard path (backward compatible).
    """
    model.gradient_checkpointing_disable()

    # Path 1: Temporal query (registry date filter)
    date_range = detect_temporal_query(text)
    if date_range:
        return _handle_temporal_query(text, date_range, history, model, tokenizer, config)

    # Path 2: Entity-routed query (targeted key probing)
    if router is not None:
        plan = router.route(text)
        if plan.strategy == "targeted_probe" and plan.steps:
            return _handle_routed_query(text, plan, history, model, tokenizer, config)

    # Path 3: Standard (facts in weights, no probing)
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
            recalled_facts.append(f"Q: {result.get('question', '')}\nA: {result.get('answer', '')}")

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
    )

    return ChatResult(text=response, probed_keys=matching_keys)


def _handle_routed_query(
    text: str,
    plan,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
) -> ChatResult:
    """Entity-routed path — probe targeted keys per adapter."""
    from paramem.models.loader import switch_adapter
    from paramem.training.indexed_memory import probe_key

    logger.info(
        "Routed query: entities=%s, %d steps",
        plan.matched_entities,
        len(plan.steps),
    )

    # Load registry for confidence verification
    registry = {}
    if config.registry_path.exists():
        with open(config.registry_path) as f:
            raw = json.load(f)
        for key, meta in raw.items():
            if isinstance(meta, dict):
                registry[key] = meta.get("simhash", 0)

    # Probe keys from each adapter in the plan
    recalled_facts = []
    probed_keys = []
    for step in plan.steps:
        # Switch to the target adapter if it exists on the model
        if hasattr(model, "peft_config") and step.adapter_name in model.peft_config:
            switch_adapter(model, step.adapter_name)
        elif step.adapter_name != "episodic":
            logger.info("Adapter %s not loaded, skipping", step.adapter_name)
            continue

        for key in step.keys_to_probe:
            result = probe_key(model, tokenizer, key, registry=registry)
            if result and "failure_reason" not in result:
                recalled_facts.append(
                    f"Q: {result.get('question', '')}\nA: {result.get('answer', '')}"
                )
                probed_keys.append(key)

    if not recalled_facts:
        logger.info(
            "No facts recalled from routed keys (entities=%s), falling back to standard path",
            plan.matched_entities,
        )
        return _handle_standard_query(text, history, model, tokenizer, config)

    # Build prompt with recalled facts as context
    facts_context = "\n\n".join(recalled_facts)
    augmented_text = (
        f"Based on the following facts:\n\n{facts_context}\n\nAnswer the question: {text}"
    )

    messages = _build_messages(augmented_text, history, config.voice.system_prompt, tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate_answer(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.3,
    )

    return ChatResult(text=response, probed_keys=probed_keys)


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
