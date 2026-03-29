"""Chat inference — tri-path routing with dual-graph matching.

Routing is based on dual-graph matching (PA knowledge graph + HA entity graph):
1. PA match → local adapter probe + base model reasoning
2. HA match → HA conversation agent (tools, device control)
3. Neither → SOTA cloud model (superior reasoning)
4. Both → PA first; if [ESCALATE], go HA (HA entities present)

Imperative detection (HA entity + action verb + non-interrogative) routes
directly to HA, skipping local inference.

Temporal queries filter keys by date range first, then follow the same
probe → reason flow.
"""

import json
import logging
from dataclasses import dataclass, field

from paramem.evaluation.recall import generate_answer
from paramem.models.loader import adapt_messages
from paramem.server.cloud.base import CloudAgent
from paramem.server.config import ServerConfig
from paramem.server.escalation import detect_escalation
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
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ChatResult:
    """Process a chat message via tri-path routing.

    Routing (dual-graph):
    1. Temporal query → filter keys by date → probe → reason
    2. Imperative + HA entity → HA agent directly (action command)
    3. PA match → probe matched keys → reason
    4. HA match (non-imperative) → HA agent (device query)
    5. No match → SOTA cloud (reasoning)
    6. Fallback → HA agent or local base model

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
                match_source="pa",
            )
            logger.info("Found %d keys for date range", len(temporal_keys))

    # Path 2: Dual-graph entity routing
    if plan is None and router is not None:
        plan = router.route(text, speaker=speaker)

    # Shared kwargs for _escalate_to_ha_agent calls
    ha_kwargs = dict(
        sota_agent=sota_agent,
        speaker=speaker,
        history=history,
    )

    # Path 2a: Imperative + HA entity → HA agent directly (action command)
    if plan and plan.imperative and plan.match_source in ("ha", "both"):
        logger.info("Imperative HA command: domains=%s", plan.ha_domains)
        sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            result = _escalate_to_ha_agent(
                sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
            )
            if result is not None:
                return result
        else:
            logger.info("Sanitizer blocked imperative HA command: %s", findings)

    # Path 2b: PA match → probe adapters and reason locally
    if plan and plan.steps and plan.match_source in ("pa", "both"):
        return _probe_and_reason(
            text,
            plan,
            history,
            model,
            tokenizer,
            config,
            cloud_agent,
            sota_agent,
            ha_client,
            tool_registry,
            speaker=speaker,
        )

    # Path 2c: HA-only match (non-imperative) → HA agent
    if plan and plan.match_source == "ha":
        logger.info("HA entity query (non-imperative): domains=%s", plan.ha_domains)
        sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            result = _escalate_to_ha_agent(
                sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
            )
            if result is not None:
                return result
        else:
            logger.info("Sanitizer blocked HA query: %s", findings)

    # Path 3: No match in either graph → SOTA cloud (reasoning)
    if sota_agent is not None:
        sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            logger.info("No graph match, routing to SOTA agent")
            return _escalate_to_sota(
                sanitized, sota_agent, config, speaker=speaker, history=history
            )
        logger.info("Sanitizer blocked SOTA query: %s", findings)

    # Path 4: Fallback — HA agent, then local base model
    if cloud_agent is not None or ha_client is not None:
        sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            logger.info("Fallback to HA agent")
            result = _escalate_to_ha_agent(
                sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
            )
            if result is not None:
                return result

    # All cloud services failed — local base model as last resort
    return _base_model_answer(
        text,
        history,
        model,
        tokenizer,
        config,
        cloud_agent=cloud_agent,
        sota_agent=sota_agent,
        ha_client=ha_client,
        tool_registry=tool_registry,
        speaker=speaker,
    )


def _escalate_to_ha_agent(
    text: str,
    ha_client: HAClient | None,
    cloud_agent: CloudAgent | None,
    config: ServerConfig,
    tool_registry: ToolRegistry | None = None,
    sota_agent: CloudAgent | None = None,
    speaker: str | None = None,
    history: list[dict] | None = None,
) -> ChatResult | None:
    """Escalate to HA conversation agent → direct Groq → SOTA.

    Returns None if all cloud services fail, so the caller can fall
    through to the local base model.
    """
    # Primary: HA conversation agent (Groq via HA)
    if ha_client is not None:
        response = ha_client.conversation_process(text, agent_id=config.ha_agent_id)
        if response is not None:
            return ChatResult(text=response, escalated=True)
        logger.warning("HA conversation.process failed, trying direct cloud")

    # Fallback 1: direct Groq call (no tools, no HA prompt)
    if cloud_agent is not None:
        result = _cloud_agent_answer(cloud_agent, text, config, tool_registry)
        if result.text and "couldn't" not in result.text:
            return result

    # Fallback 2: SOTA agent
    if sota_agent is not None:
        logger.info("HA/Groq failed, trying SOTA")
        return _escalate_to_sota(text, sota_agent, config, speaker=speaker, history=history)

    # All cloud services down — return None so caller can use local model
    logger.warning("All cloud services failed")
    return None


CLOUD_DIRECT_PROMPT = (
    "You are a helpful personal assistant. Be concise. "
    "Answer in 1-2 spoken sentences. Do not use markdown, lists, or structured formatting. "
)

SOTA_PROMPT = (
    "You are continuing a conversation as a personal assistant. "
    "Derive your persona, tone, and conversational style from the "
    "preceding conversation. Answer clearly and concisely in 1-3 spoken "
    "sentences. Do not use markdown, lists, or structured formatting."
)


def _sanitize_history(
    history: list[dict] | None,
    mode: str,
) -> list[dict]:
    """Sanitize conversation history for cloud, dropping blocked turns."""
    if not history:
        return []
    sanitized = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        if not text:
            continue
        clean, _ = sanitize_for_cloud(text, mode=mode)
        if clean is not None:
            sanitized.append({"role": role, "text": clean})
    return sanitized


def _escalate_to_sota(
    text: str,
    sota_agent: CloudAgent,
    config: ServerConfig,
    speaker: str | None = None,
    history: list[dict] | None = None,
) -> ChatResult:
    """Route to SOTA cloud model for reasoning-heavy queries.

    Passes sanitized conversation history so the SOTA model can derive
    persona, tone, and style from the conversation context. Personal
    details are stripped by the sanitizer — only the conversational
    pattern survives.
    """
    sanitized_history = _sanitize_history(history, mode=config.sanitization.mode)

    # Build system prompt with speaker name if available
    prompt = SOTA_PROMPT
    if speaker:
        prompt = f"You are speaking with {speaker}. " + prompt

    logger.info(
        "SOTA escalation (%d history turns): %s",
        len(sanitized_history),
        text[:100],
    )
    response = sota_agent.call(
        query=text,
        system_prompt=prompt,
        history=sanitized_history,
    )
    if response.text:
        return ChatResult(text=response.text, escalated=True)
    return ChatResult(text="I couldn't get an answer right now.", escalated=True)


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
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
    speaker: str | None = None,
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
        logger.info("All probed keys failed, falling back (match_source=%s)", plan.match_source)
        sanitized, findings = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            ha_kwargs = dict(
                sota_agent=sota_agent,
                speaker=speaker,
                history=history,
            )
            # Context-aware fallback: HA entities present → HA chain, otherwise → SOTA
            if plan.match_source == "both":
                result = _escalate_to_ha_agent(
                    sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
                )
                if result is not None:
                    return result
            elif sota_agent is not None:
                return _escalate_to_sota(
                    sanitized, sota_agent, config, speaker=speaker, history=history
                )
            else:
                result = _escalate_to_ha_agent(
                    sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
                )
                if result is not None:
                    return result
        return _base_model_answer(
            text,
            history,
            model,
            tokenizer,
            config,
            cloud_agent=cloud_agent,
            sota_agent=sota_agent,
            ha_client=ha_client,
            tool_registry=tool_registry,
            speaker=speaker,
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
        match_source=plan.match_source,
        probed_keys=successful_keys,
        cloud_agent=cloud_agent,
        sota_agent=sota_agent,
        ha_client=ha_client,
        tool_registry=tool_registry,
        speaker=speaker,
        history=history,
    )


def _base_model_answer(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    cloud_agent: CloudAgent | None = None,
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
    speaker: str | None = None,
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
        sota_agent=sota_agent,
        ha_client=ha_client,
        tool_registry=tool_registry,
        speaker=speaker,
        history=history,
    )


def _maybe_escalate(
    response: str,
    config: ServerConfig,
    match_source: str = "none",
    probed_keys: list[str] | None = None,
    cloud_agent: CloudAgent | None = None,
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    tool_registry: ToolRegistry | None = None,
    speaker: str | None = None,
    history: list[dict] | None = None,
) -> ChatResult:
    """Check for [ESCALATE] tag and route based on match_source.

    Context-aware escalation:
    - HA entities present (match_source "both"/"ha") → HA agent
    - No HA entities (match_source "pa"/"none") → SOTA agent
    - Fallback: HA agent, then strip tag
    """
    should_escalate, forwarded_query = detect_escalation(response)

    if should_escalate:
        sanitized, _ = sanitize_for_cloud(forwarded_query, mode=config.sanitization.mode)
        if sanitized is None:
            should_escalate = False
        else:
            # Context-aware: route based on which graph matched.
            # _escalate_to_ha_agent handles the full chain: HA → Groq → SOTA.
            ha_kwargs = dict(
                sota_agent=sota_agent,
                speaker=speaker,
                history=history,
            )
            if match_source in ("both", "ha"):
                logger.info("[ESCALATE] → HA (source=%s): %s", match_source, sanitized[:100])
                result = _escalate_to_ha_agent(
                    sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
                )
                if result is not None:
                    return result
            elif sota_agent is not None:
                logger.info("[ESCALATE] → SOTA (source=%s): %s", match_source, sanitized[:100])
                return _escalate_to_sota(
                    sanitized, sota_agent, config, speaker=speaker, history=history
                )
            else:
                logger.info("[ESCALATE] → HA fallback: %s", sanitized[:100])
                result = _escalate_to_ha_agent(
                    sanitized, ha_client, cloud_agent, config, tool_registry, **ha_kwargs
                )
                if result is not None:
                    return result

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
