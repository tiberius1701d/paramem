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
from paramem.server.config import ServerConfig
from paramem.server.escalation import detect_escalation, escalate_to_cloud
from paramem.server.router import RoutingPlan, RoutingStep
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
    speaker: str | None,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    router=None,
) -> ChatResult:
    """Process a chat message via entity-routed inference.

    The speaker entity is always injected into routing so that personal
    queries ("What is my name?") resolve to the speaker's keys even
    when no explicit entity appears in the query text.

    Temporal queries filter keys by date range first, then probe.
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
            # Wrap temporal keys as a single episodic step
            plan = RoutingPlan(
                steps=[RoutingStep(adapter_name="episodic", keys_to_probe=temporal_keys)],
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
        return _probe_and_reason(text, plan, history, model, tokenizer, config)

    # No keys found — base model without context (escalation candidate)
    return _base_model_answer(text, history, model, tokenizer, config)


def _probe_and_reason(
    text: str,
    plan: RoutingPlan,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
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
                layer_facts.append(
                    f"- {result.get('answer', '')}"
                )
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
        logger.info("All probed keys failed, falling back to base model")
        return _base_model_answer(text, history, model, tokenizer, config)

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
    augmented_text = (
        f"What you know about the speaker:\n\n{layered_context}\n\n"
        f"Question: {text}"
    )

    messages = _build_messages(augmented_text, history, config.voice.load_prompt(), tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            response = generate_answer(
                model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
            )
    else:
        response = generate_answer(
            model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
        )

    return _maybe_escalate(response, config, probed_keys=successful_keys)


def _base_model_answer(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
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
        response = generate_answer(
            model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
        )

    return _maybe_escalate(response, config)


def _maybe_escalate(
    response: str,
    config: ServerConfig,
    probed_keys: list[str] | None = None,
) -> ChatResult:
    """Check for [ESCALATE] tag and forward to cloud if configured."""
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
