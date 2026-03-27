"""Chat inference — entity-routed path with speaker injection.

All personal queries go through the same path:
1. Inject speaker entity + extract query entities → find keys via router
2. For each adapter in the routing plan: switch adapter, probe its keys
3. Adapter OFF → feed all recalled facts as context → base model reasons

Temporal queries filter keys by date range first, then follow the same
probe → reason flow.

There is no "standard" path — if no keys are found, the base model
answers without context (escalation candidate).
"""

import json
import logging
from dataclasses import dataclass, field

from paramem.evaluation.recall import generate_answer
from paramem.models.loader import adapt_messages
from paramem.server.cloud.base import CloudAgent
from paramem.server.config import ServerConfig
from paramem.server.escalation import detect_escalation, escalate_to_cloud
from paramem.server.router import RoutingPlan, RoutingStep
from paramem.server.sanitizer import sanitize_for_cloud
from paramem.server.temporal import detect_temporal_query, filter_registry_by_date
from paramem.server.tools.ha_client import HAClient
from paramem.server.tools.registry import ToolRegistry

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
    speaker: str | None,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    router=None,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Process a chat message via entity-routed inference.

    Routing:
    1. Temporal query → filter keys by date → probe → reason
    2. Entity match → probe matched keys → reason
    3. No match + cloud configured → direct to cloud (no local inference)
    4. No match + no cloud → local base model (may [ESCALATE])

    The speaker entity is always injected into routing so that personal
    queries ("What is my name?") resolve to the speaker's keys even
    when no explicit entity appears in the query text.
    """
    model.gradient_checkpointing_disable()

    # Build a routing plan
    plan = None

    # Path 1: Temporal query — filter keys by date range
    date_range = detect_temporal_query(text)
    if date_range:
        start_date, end_date = date_range
        logger.info("Temporal query detected: %s to %s", start_date, end_date)
        temporal_keys = filter_registry_by_date(config.registry_path, start_date, end_date)
        if temporal_keys:
            plan = RoutingPlan(
                steps=[
                    RoutingStep(
                        adapter_name="episodic",
                        keys_to_probe=temporal_keys,
                    )
                ],
                strategy="temporal",
            )
            logger.info("Found %d keys for date range", len(temporal_keys))

    # Path 2: Entity routing (always, unless temporal already found keys)
    if plan is None and router is not None:
        plan = router.route(text, speaker=speaker)
        if plan.steps:
            logger.info(
                "Routed query: entities=%s, %d steps",
                plan.matched_entities,
                len(plan.steps),
            )

    # Probe keys per adapter → reconstruct facts → reason
    if plan and plan.steps:
        return _probe_and_reason(
            text,
            plan,
            history,
            model,
            tokenizer,
            config,
            cloud_agent,
            ha_client,
            tool_registry,
        )

    # Path 3: No entity match — direct to cloud (skip local inference)
    if cloud_agent is not None:
        sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is None:
            logger.info("Cloud blocked by sanitizer: %s", findings)
            return _base_model_answer(
                text,
                history,
                model,
                tokenizer,
                config,
                cloud_agent=cloud_agent,
                ha_client=ha_client,
                tool_registry=tool_registry,
            )
        logger.info("No entity match, routing directly to cloud agent")
        return _escalate_to_ha_agent(sanitized, ha_client, cloud_agent, config, tool_registry)

    # Path 4: No cloud — local base model (may emit [ESCALATE])
    return _base_model_answer(
        text,
        history,
        model,
        tokenizer,
        config,
        cloud_agent=cloud_agent,
        ha_client=ha_client,
        tool_registry=tool_registry,
    )


def _escalate_to_ha_agent(
    text: str,
    ha_client: HAClient | None,
    cloud_agent: CloudAgent | None,
    config: ServerConfig,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Escalate a query to HA's conversation agent, with direct Groq fallback.

    Primary path: forward to HA's conversation.process which handles
    prompt rendering, tool execution, and entity resolution internally.
    Fallback: call Groq directly if HA is unavailable.
    """
    # Primary: HA conversation agent
    if ha_client is not None:
        response = ha_client.conversation_process(text, agent_id=config.ha_agent_id)
        if response is not None:
            return ChatResult(text=response, escalated=True)
        logger.warning("HA conversation.process failed, falling back to direct cloud")

    # Fallback: direct Groq call (no tools, no HA prompt)
    if cloud_agent is not None:
        return _cloud_agent_answer(cloud_agent, text, config, tool_registry)

    return ChatResult(
        text="I couldn't get an answer right now.",
        escalated=True,
    )


CLOUD_DIRECT_PROMPT = (
    "You are a helpful personal assistant. Be concise. "
    "Answer in 1-2 spoken sentences. Do not use markdown, lists, or structured formatting. "
)


def _cloud_agent_answer(
    cloud_agent: CloudAgent,
    text: str,
    config: ServerConfig,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Direct cloud agent call — fallback when HA is unavailable.

    Simple query-response without HA tools or prompt context.
    """
    response = cloud_agent.call(
        query=text,
        system_prompt=CLOUD_DIRECT_PROMPT,
    )

    if response.text:
        return ChatResult(text=response.text, escalated=True)

    return ChatResult(
        text="I'm sorry, I couldn't get an answer right now.",
        escalated=True,
    )


def _probe_and_reason(
    text: str,
    plan: RoutingPlan,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Probe adapters in memory hierarchy order, assemble layered context.

    Replicates the intended adapter switching architecture into the
    context window. Each adapter's recalled facts become a named section,
    preserving the query order:

        [Procedural]  — behavioral preferences, response style
        [Semantic]    — consolidated stable knowledge
        [Episodic]    — recent conversational knowledge

    For each adapter:
      1. Switch to it
      2. Probe its keys
      3. Collect recalled facts into that layer's section

    After all adapters are probed, disable adapters and let the base
    model reason over the layered context.
    """
    from peft import PeftModel

    from paramem.models.loader import switch_adapter
    from paramem.training.indexed_memory import probe_key

    # Load registry for SimHash verification
    registry = _load_simhash_registry(config.registry_path)

    # Section labels for the layered context — maps adapter name to its role
    LAYER_LABELS = {
        "procedural": "Behavioral preferences",
        "semantic": "Consolidated knowledge",
        "episodic": "Recent knowledge",
    }

    # Probe each adapter's keys sequentially, collecting per-layer facts
    layers: dict[str, list[str]] = {}
    successful_keys = []

    for step in plan.steps:
        # Switch to the target adapter if it exists on the model
        if hasattr(model, "peft_config") and step.adapter_name in model.peft_config:
            switch_adapter(model, step.adapter_name)
        elif step.adapter_name != "episodic":
            logger.info("Adapter %s not loaded, skipping", step.adapter_name)
            continue

        layer_facts = []
        for key in step.keys_to_probe:
            result = probe_key(model, tokenizer, key, registry=registry)
            if result and "failure_reason" not in result:
                layer_facts.append(f"- {result.get('answer', '')}")
                successful_keys.append(key)

        if layer_facts:
            layers[step.adapter_name] = layer_facts

        logger.info(
            "Adapter %s: probed %d keys, recalled %d facts",
            step.adapter_name,
            len(step.keys_to_probe),
            len(layer_facts),
        )

    if not layers:
        logger.info("All probed keys failed, falling back")
        if cloud_agent is not None:
            sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
            if sanitized is not None:
                return _escalate_to_ha_agent(
                    sanitized, ha_client, cloud_agent, config, tool_registry
                )
            logger.info("Cloud blocked by sanitizer on fallback: %s", findings)
        return _base_model_answer(
            text,
            history,
            model,
            tokenizer,
            config,
            cloud_agent=cloud_agent,
            ha_client=ha_client,
            tool_registry=tool_registry,
        )

    total_facts = sum(len(f) for f in layers.values())
    logger.info("Total recalled: %d facts from %d layers", total_facts, len(layers))

    # Assemble layered context in hierarchy order
    context_sections = []
    for adapter_name in ["procedural", "semantic", "episodic"]:
        if adapter_name in layers:
            label = LAYER_LABELS.get(adapter_name, adapter_name)
            facts_text = "\n".join(layers[adapter_name])
            context_sections.append(f"[{label}]\n{facts_text}")

    layered_context = "\n\n".join(context_sections)
    augmented_text = f"What you know about the speaker:\n\n{layered_context}\n\nQuestion: {text}"

    messages = _build_messages(augmented_text, history, config.voice.load_prompt(), tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            response = generate_answer(
                model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
            )
    else:
        response = generate_answer(model, tokenizer, prompt, max_new_tokens=256, temperature=0.0)

    return _maybe_escalate(
        response,
        config,
        probed_keys=successful_keys,
        cloud_agent=cloud_agent,
        ha_client=ha_client,
        tool_registry=tool_registry,
    )


def _base_model_answer(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Answer from base model without context — escalation candidate."""
    from peft import PeftModel

    messages = _build_messages(text, history, config.voice.load_prompt(), tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            response = generate_answer(
                model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
            )
    else:
        response = generate_answer(model, tokenizer, prompt, max_new_tokens=256, temperature=0.0)

    return _maybe_escalate(
        response,
        config,
        cloud_agent=cloud_agent,
        ha_client=ha_client,
        tool_registry=tool_registry,
    )


def _maybe_escalate(
    response: str,
    config: ServerConfig,
    probed_keys: list[str] | None = None,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Check for [ESCALATE] tag and forward to HA/cloud if configured.

    Secondary escalation path — only fires when the graph falsely routes
    to memory (entity match but model can't answer). Delegates to
    _escalate_to_ha_agent (HA conversation agent primary, direct cloud fallback).
    """
    should_escalate, forwarded_query = detect_escalation(response)

    if should_escalate and cloud_agent is not None:
        sanitized, _ = sanitize_for_cloud(forwarded_query, mode=config.sanitization.mode)
        if sanitized is None:
            should_escalate = False
        else:
            logger.info("Escalating to cloud: %s", sanitized[:100])
            return _escalate_to_ha_agent(sanitized, ha_client, cloud_agent, config, tool_registry)
    elif should_escalate and config.general_agent.enabled:
        # Legacy path — direct HTTP call without adapter
        sanitized_legacy, _ = sanitize_for_cloud(forwarded_query, mode=config.sanitization.mode)
        if sanitized_legacy is None:
            should_escalate = False
        else:
            logger.info("Escalating to cloud (legacy): %s", sanitized_legacy[:100])
            cloud_response = escalate_to_cloud(sanitized_legacy, config.general_agent)
            if cloud_response:
                return ChatResult(text=cloud_response, escalated=True)

    if should_escalate:
        response = response.replace("[ESCALATE]", "").strip()
        if not response:
            response = "I'm not sure about that."

    return ChatResult(text=response, probed_keys=probed_keys or [])


def _load_simhash_registry(registry_path) -> dict:
    """Load SimHash values from registry for probe verification."""
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            raw = json.load(f)
        for key, meta in raw.items():
            if isinstance(meta, dict):
                registry[key] = meta.get("simhash", 0)
    return registry


def _build_messages(
    text: str,
    history: list[dict] | None,
    system_prompt: str,
    tokenizer,
) -> list[dict]:
    """Build chat messages from user text, history, and system prompt.

    Enforces strict user/assistant alternation required by Mistral and
    other chat templates. Consecutive same-role messages are merged.
    """
    # Build alternating user/assistant pairs from history.
    # Mistral requires: system → user → assistant → user → ...
    # HA may send non-alternating history, so we enforce the pattern.
    pairs = []
    if history:
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role", "user")
            content = turn.get("text", "")
            if role in ("user", "assistant") and content:
                pairs.append({"role": role, "content": content})

    # Merge consecutive same-role messages
    merged = []
    for msg in pairs:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    # Strip leading assistant messages (must start with user after system)
    while merged and merged[0]["role"] == "assistant":
        merged.pop(0)

    messages = [{"role": "system", "content": system_prompt}] + merged

    # Ensure the final message is a user turn with the current text
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n" + text
    else:
        messages.append({"role": "user", "content": text})

    return adapt_messages(messages, tokenizer)
