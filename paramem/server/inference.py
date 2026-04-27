"""Chat inference — tri-path routing with dual-graph matching.

Routing is based on dual-graph matching (PA knowledge graph + HA entity graph):
1. PA match → local adapter probe + base model reasoning
2. HA match → HA conversation agent (tools, device control)
3. Neither → HA first (tools, real-time data), SOTA fallback (reasoning)
4. [ESCALATE] from local model → HA first (tools), SOTA fallback (reasoning)

Imperative detection (HA entity + action verb + non-interrogative) routes
directly to HA, skipping local inference.

Temporal queries filter keys by date range first, then follow the same
probe → reason flow.

Fallback chain at every escalation point: HA → SOTA → local base model.
_escalate_to_ha_agent is HA-only; callers own the SOTA fallback.
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

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


def enqueue_post_session_train(
    conversation_id: str,
    transcript: str,
    speaker_id: str,
    speaker_name: str | None,
    loop,
    background_trainer,
    config: "ServerConfig",
    state: dict,
    post_session_queue=None,
) -> None:
    """Enqueue a post-conversation training job on the BackgroundTrainer.

    Submits a callable wrapping ``loop.post_session_train`` to
    ``BackgroundTrainer.submit()``.  The single-slot worker thread holds the
    GPU lock for the duration of each job, so jobs from concurrent ``/chat``
    turns queue behind one another and execute serially.  This eliminates
    concurrent-turn races on shared ``ConsolidationLoop`` state
    (``_indexed_next_index``).

    ``inference_fallback_adapter`` is set to ``"episodic"`` (the stable main
    adapter) so that inference during a paused training job reads committed
    weights rather than mid-training staging state.

    After ``post_session_train`` returns, ``state["model"]`` is updated with
    ``loop.model`` to pick up any PeftModel handle rebinding that
    ``create_interim_adapter`` may have performed.

    This function is intentionally side-effect free on the conversation path —
    it never blocks the response being returned to the caller.

    When *post_session_queue* is provided the entry for *conversation_id* is
    removed from the persistent queue on successful completion of
    ``post_session_train``.  The caller is responsible for calling
    ``post_session_queue.enqueue(...)`` **before** calling this function so
    that a crash between enqueue and training-start is recoverable on the next
    server restart.

    Args:
        conversation_id: The conversation identifier (used as ``session_id``).
        transcript: The full conversation transcript text (all turns joined).
        speaker_id: Speaker identifier for key ownership scoping.
        speaker_name: Human-readable speaker name for extraction personalisation.
        loop: The live ``ConsolidationLoop`` instance (``_state["consolidation_loop"]``).
        background_trainer: The live ``BackgroundTrainer`` instance
            (``_state["background_trainer"]``).  Must not be ``None`` — callers
            should guard with ``background_trainer is not None`` before calling.
        config: Server config for ``schedule`` and ``max_interim_count``.
        state: Global ``_state`` dict; ``state["model"]`` is updated after
            training completes.
        post_session_queue: Optional :class:`~paramem.server.post_session_queue.PostSessionQueue`
            instance.  When provided, a successful training run removes
            *conversation_id* from the queue so it is not replayed on the next
            server restart.
    """
    if loop is None or background_trainer is None:
        return

    schedule = config.consolidation.refresh_cadence
    max_interim_count = config.consolidation.max_interim_count

    def _run_post_session_train() -> None:
        """Execute post_session_train and update state. Runs in BackgroundTrainer worker."""
        try:
            result = loop.post_session_train(
                session_transcript=transcript,
                session_id=conversation_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                schedule=schedule,
                max_interim_count=max_interim_count,
            )
            # After post_session_train returns, update global model handle in case
            # create_interim_adapter rebound the PeftModel wrapper.
            state["model"] = loop.model
            logger.info(
                "post_session_train complete: mode=%s, adapter=%s, new_keys=%d",
                result.get("mode"),
                result.get("adapter_name"),
                len(result.get("new_keys", [])),
            )
            # Remove the entry from the persistent queue on success so it is
            # not replayed on the next server restart.
            if post_session_queue is not None:
                post_session_queue.remove(conversation_id)
        except Exception:
            logger.exception("post_session_train failed for conversation %s", conversation_id)
            # Entry intentionally left in queue on failure so a future restart
            # can retry.

    background_trainer.submit(
        _run_post_session_train,
        inference_fallback_adapter="episodic",
    )


def _language_instruction(language: str | None, config: ServerConfig | None = None) -> str:
    """Return a language instruction string, or empty for English/unknown.

    Derives the display name from TTS config (voice language_name field),
    falling back to ISO 639 standard names.
    """
    if not language or language == "en":
        return ""
    if config is not None:
        name = config.tts.language_name(language)
    else:
        from paramem.server.config import ISO_LANGUAGE_NAMES

        name = ISO_LANGUAGE_NAMES.get(language, language)
    return f"Respond in {name}."


def _personalize_prompt(
    base_prompt: str,
    speaker: str | None,
    language: str | None = None,
    config: ServerConfig | None = None,
) -> str:
    """Inject speaker name and language instruction into the system prompt.

    Greeting is handled at the app layer (prepended to response text)
    so it works across all paths including escalation.
    """
    parts = []
    if speaker:
        parts.append(f"You are speaking with {speaker}.")
    lang_instr = _language_instruction(language, config)
    if lang_instr:
        parts.append(lang_instr)
    prefix = " ".join(parts)
    if prefix:
        return prefix + " " + base_prompt
    return base_prompt


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
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    known_entities: set[str] | None = None,
) -> ChatResult:
    """Process a chat message via tri-path routing.

    Routing (dual-graph):
    1. Temporal query → filter keys by date → probe → reason
    2. Imperative + HA entity → HA agent directly (action command)
    3. PA match → probe matched keys → reason
    4. HA match (non-imperative) → HA agent (device query)
    5. No match → HA first (tools, real-time data), SOTA fallback (reasoning)
    6. All cloud failed → local base model

    When ``config.debug`` is True a per-request routing-decision diagnostic
    is emitted via ``logging.info(extra={"routing": …})`` at function exit.
    The dict captures which paths were attempted, which one returned, and
    whether the request was a "residual" (no graph match, no HA match) —
    the metric the residual classifier work is gated on.  Off when
    ``config.debug`` is False to keep production logs lean.
    """
    routing_diags: dict = {
        "conversation_id": conversation_id,
        "match_source": "none",
        "imperative": False,
        "paths_attempted": [],
        "fallthrough_reason": None,
        "exit_via": None,
        "is_residual": False,
    }
    try:
        model.gradient_checkpointing_disable()

        plan = None
        allowed_keys = None
        if speaker_id and router is not None:
            allowed_keys = router._speaker_key_index.get(speaker_id, set())

        # Path 1: Temporal query — filter keys by date range
        date_range = detect_temporal_query(text)
        if date_range:
            start_date, end_date = date_range
            logger.info("Temporal query detected: %s to %s", start_date, end_date)
            temporal_keys = filter_registry_by_date(config.registry_path, start_date, end_date)
            if allowed_keys is not None:
                temporal_keys = [k for k in temporal_keys if k in allowed_keys]
            if temporal_keys:
                plan = RoutingPlan(
                    steps=[RoutingStep(adapter_name="episodic", keys_to_probe=temporal_keys)],
                    strategy="temporal",
                    match_source="pa",
                )
                routing_diags["paths_attempted"].append("temporal")
                logger.info("Found %d keys for date range", len(temporal_keys))

        # Path 2: Dual-graph entity routing
        if plan is None and router is not None:
            plan = router.route(text, speaker=speaker, speaker_id=speaker_id)
        if plan is not None:
            routing_diags["match_source"] = plan.match_source
            routing_diags["imperative"] = plan.imperative

        # Pre-compute sanitization once for all cloud escalation paths.
        # Personal-content detection is anchored on the router's entity index
        # (the graph's ground truth) plus a first-person token-set + the
        # resolved speaker_id — the same ground truth the extraction-path
        # anonymizer uses, no static keyword list.
        if known_entities is None and router is not None and hasattr(router, "_all_entities"):
            known_entities = router._all_entities
        sanitized_text, sanitization_findings = sanitize_for_cloud(
            text,
            mode=config.sanitization.mode,
            speaker_id=speaker_id,
            known_entities=known_entities,
        )

        # Path 2a: Imperative + HA entity → HA agent directly (action command)
        if plan and plan.imperative and plan.match_source in ("ha", "both"):
            routing_diags["paths_attempted"].append("2a")
            logger.info("Imperative HA command: domains=%s", plan.ha_domains)
            if sanitized_text is not None:
                result = _escalate_to_ha_agent(sanitized_text, ha_client, config, language=language)
                if result is not None:
                    routing_diags["exit_via"] = "2a_ha"
                    return result
                # Only fall to SOTA when there is no PA match to probe in path 2b
                if sota_agent is not None and plan.match_source == "ha":
                    routing_diags["exit_via"] = "2a_sota"
                    return _escalate_to_sota(
                        sanitized_text,
                        sota_agent,
                        config,
                        speaker=speaker,
                        history=history,
                        language=language,
                    )
            else:
                routing_diags["fallthrough_reason"] = "sanitizer_blocked_2a"
                logger.info("Sanitizer blocked imperative HA command: %s", sanitization_findings)

        # Path 2b: PA match → probe adapters and reason locally
        if plan and plan.steps and plan.match_source in ("pa", "both"):
            routing_diags["paths_attempted"].append("2b")
            routing_diags["exit_via"] = "2b_pa_probe"
            return _probe_and_reason(
                text,
                plan,
                history,
                model,
                tokenizer,
                config,
                sota_agent=sota_agent,
                ha_client=ha_client,
                speaker=speaker,
                language=language,
            )

        # Path 2c: HA-only match (non-imperative) → HA agent
        if plan and plan.match_source == "ha":
            routing_diags["paths_attempted"].append("2c")
            logger.info("HA entity query (non-imperative): domains=%s", plan.ha_domains)
            if sanitized_text is not None:
                result = _escalate_to_ha_agent(sanitized_text, ha_client, config, language=language)
                if result is not None:
                    routing_diags["exit_via"] = "2c_ha"
                    return result
                if sota_agent is not None:
                    routing_diags["exit_via"] = "2c_sota"
                    return _escalate_to_sota(
                        sanitized_text,
                        sota_agent,
                        config,
                        speaker=speaker,
                        history=history,
                        language=language,
                    )
            else:
                routing_diags["fallthrough_reason"] = "sanitizer_blocked_2c"
                logger.info("Sanitizer blocked HA query: %s", sanitization_findings)

        # Path 3: No match → HA first (has tools for real-time data), SOTA fallback
        if sanitized_text is not None:
            routing_diags["paths_attempted"].append("3")
            logger.info("No graph match, trying HA agent (tools)")
            result = _escalate_to_ha_agent(sanitized_text, ha_client, config, language=language)
            if result is not None:
                routing_diags["exit_via"] = "3_ha"
                return result
            if sota_agent is not None:
                routing_diags["exit_via"] = "3_sota"
                logger.info("HA failed, routing to SOTA agent")
                return _escalate_to_sota(
                    sanitized_text,
                    sota_agent,
                    config,
                    speaker=speaker,
                    history=history,
                    language=language,
                )
        else:
            routing_diags["fallthrough_reason"] = "sanitizer_blocked_3"
            logger.info("Sanitizer blocked query: %s", sanitization_findings)

        # Abstention: self-referential query with no local match → canned response.
        # The bare base model would otherwise confabulate personal data here
        # (e.g. "Where do I live?" → "New York City" on an untrained adapter).
        # Personal-claim / possessive findings in *statements* (introductions,
        # fact-sharing) are not a confabulation risk — the user is the source of
        # the facts in the same turn — so they fall through to the base model
        # for conversational acknowledgement.
        #
        # Two response variants distinguish the two states:
        #
        # * Cold start — speaker is identified but the router has no keys for
        #   them yet (typical between enrollment and the next consolidation).
        #   The canned "I don't have that information stored yet" reads as
        #   confused in that state because the system *can't* have facts about
        #   a freshly enrolled speaker. Use ``cold_start_response`` instead.
        # * Coverage gap — speaker has parametric facts but this query missed.
        #   The standard ``response`` is appropriate.
        if (
            sanitized_text is None
            and config.abstention.enabled
            and "self_referential" in sanitization_findings
        ):
            is_cold_start = bool(speaker_id) and (
                router is None or not router._speaker_key_index.get(speaker_id)
            )
            response_text = (
                config.abstention.load_cold_start_response()
                if is_cold_start
                else config.abstention.load_response()
            )
            routing_diags["paths_attempted"].append("abstention")
            routing_diags["exit_via"] = (
                "abstention_cold_start" if is_cold_start else "abstention_canned"
            )
            logger.info(
                "Abstention: self-referential query + no local match (cold_start=%s)",
                is_cold_start,
            )
            return ChatResult(text=response_text)

        # All cloud services failed — local base model as last resort
        routing_diags["paths_attempted"].append("base")
        routing_diags["exit_via"] = "base_model"
        return _base_model_answer(
            text,
            history,
            model,
            tokenizer,
            config,
            sota_agent=sota_agent,
            ha_client=ha_client,
            speaker=speaker,
            language=language,
        )
    finally:
        if getattr(config, "debug", False):
            routing_diags["is_residual"] = routing_diags["match_source"] == "none"
            logger.info("routing decision", extra={"routing": routing_diags})


def _escalate_to_ha_agent(
    text: str,
    ha_client: HAClient | None,
    config: ServerConfig,
    language: str | None = None,
) -> ChatResult | None:
    """Forward to the HA conversation agent.

    Returns None if HA is unavailable or the request fails. Callers own
    the SOTA fallback — this function is HA-only.
    """
    if ha_client is None:
        logger.debug("HA escalation skipped — ha_client not configured")
        return None
    ha_languages = config.tools.ha.supported_languages if config else []
    response = ha_client.conversation_process(
        text,
        agent_id=config.ha_agent_id,
        language=language,
        supported_languages=ha_languages,
    )
    if response is not None:
        return ChatResult(text=response, escalated=True)
    logger.warning("HA conversation.process failed")
    return None


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
    language: str | None = None,
) -> ChatResult:
    """Route to SOTA cloud model for reasoning-heavy queries.

    Passes sanitized conversation history so the SOTA model can derive
    persona, tone, and style from the conversation context. Personal
    details are stripped by the sanitizer — only the conversational
    pattern survives.
    """
    sanitized_history = _sanitize_history(history, mode=config.sanitization.mode)

    prompt = SOTA_PROMPT
    parts = []
    if speaker:
        parts.append(f"You are speaking with {speaker}.")
    lang_instr = _language_instruction(language, config)
    if lang_instr:
        parts.append(lang_instr)
    if parts:
        prompt = " ".join(parts) + " " + prompt

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


def _probe_and_reason(
    text: str,
    plan: RoutingPlan,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    language: str | None = None,
) -> ChatResult:
    """Probe adapters in memory hierarchy order, assemble layered context.

    Builds a ``keys_by_adapter`` dict from ``plan.steps`` (preserving router
    order: procedural → episodic → semantic → session adapters newest-first),
    dispatches to ``probe_keys_grouped_by_adapter`` for a single
    ``switch_adapter`` call per adapter group, then reassembles per-layer
    facts for context augmentation.

    After probing, restores the model to the ``episodic`` adapter so the next
    query starts from a predictable state. The reasoning phase uses
    ``model.disable_adapter()`` so the active adapter during generation does
    not matter — only the post-return state (restored here) does.
    """
    from peft import PeftModel

    from paramem.models.loader import switch_adapter
    from paramem.training.indexed_memory import (
        probe_keys_from_disk,
        probe_keys_grouped_by_adapter,
    )

    registry = _load_simhash_registry(config.registry_path)

    LAYER_LABELS = {
        "procedural": "Behavioral preferences",
        "semantic": "Consolidated knowledge",
        "episodic": "Recent knowledge",
    }

    # Build ordered keys_by_adapter dict from routing steps.
    # Insertion order matches router output (procedural → episodic → semantic
    # → session adapters newest-first).  Use a plain dict — Python 3.7+
    # guarantees insertion-order preservation.
    keys_by_adapter: dict[str, list[str]] = {}
    for step in plan.steps:
        keys_by_adapter[step.adapter_name] = list(step.keys_to_probe)

    # Simulate mode: recall from disk-persisted keyed_pairs.json instead of
    # probing adapter weights. Blackbox-equivalent under perfect recall.
    if config.consolidation.mode == "simulate":
        probe_results = probe_keys_from_disk(config.adapter_dir, keys_by_adapter)
    else:
        # One switch_adapter call per adapter group.
        probe_results = probe_keys_grouped_by_adapter(
            model,
            tokenizer,
            keys_by_adapter,
            registry=registry,
        )

        # Restore predictable adapter state: episodic is the main adapter for
        # PM inference. The reasoning phase uses disable_adapter() so this only
        # matters for subsequent queries, not the current one.
        if hasattr(model, "peft_config") and "episodic" in model.peft_config:
            switch_adapter(model, "episodic")

    # Reassemble per-step facts so each adapter's results go to its layer.
    layers: dict[str, list[str]] = {}
    successful_keys = []

    for step in plan.steps:
        layer_facts = []
        for key in step.keys_to_probe:
            result = probe_results.get(key)
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
        logger.info(
            "All %d probed key(s) failed, escalating via HA → SOTA (source=%s)",
            sum(len(s.keys_to_probe) for s in plan.steps),
            plan.match_source,
        )
        sanitized, _ = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            result = _escalate_to_ha_agent(sanitized, ha_client, config, language=language)
            if result is not None:
                return result
            if sota_agent is not None:
                return _escalate_to_sota(
                    sanitized,
                    sota_agent,
                    config,
                    speaker=speaker,
                    history=history,
                    language=language,
                )
        return _base_model_answer(
            text,
            history,
            model,
            tokenizer,
            config,
            sota_agent=sota_agent,
            ha_client=ha_client,
            speaker=speaker,
            language=language,
        )

    total_facts = sum(len(f) for f in layers.values())
    logger.info("Total recalled: %d facts from %d layers", total_facts, len(layers))

    # Assemble layered context: procedural → episodic → semantic.
    # Later sections sit closer to the query, giving them higher recency bias.
    context_sections = []
    for adapter_name in ["procedural", "episodic", "semantic"]:
        if adapter_name in layers:
            label = LAYER_LABELS.get(adapter_name, adapter_name)
            facts_text = "\n".join(layers[adapter_name])
            context_sections.append(f"[{label}]\n{facts_text}")

    layered_context = "\n\n".join(context_sections)
    augmented_text = f"What you know about the speaker:\n\n{layered_context}\n\nQuestion: {text}"

    system_prompt = _personalize_prompt(config.voice.load_prompt(), speaker, language, config)
    messages = _build_messages(augmented_text, history, system_prompt, tokenizer)
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
        sota_agent=sota_agent,
        ha_client=ha_client,
        speaker=speaker,
        history=history,
        language=language,
    )


def _base_model_answer(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    language: str | None = None,
) -> ChatResult:
    """Answer from base model without context — escalation candidate."""
    from peft import PeftModel

    system_prompt = _personalize_prompt(config.voice.load_prompt(), speaker, language, config)
    messages = _build_messages(text, history, system_prompt, tokenizer)
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
        sota_agent=sota_agent,
        ha_client=ha_client,
        speaker=speaker,
        history=history,
        language=language,
    )


def _maybe_escalate(
    response: str,
    config: ServerConfig,
    match_source: str = "none",
    probed_keys: list[str] | None = None,
    sota_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    history: list[dict] | None = None,
    language: str | None = None,
) -> ChatResult:
    """Check for [ESCALATE] tag and route HA → SOTA.

    HA agent has tools (search, device control, real-time data) so it
    gets first shot. SOTA handles queries that need pure reasoning.
    When all escalation paths fail, the pre-escalation portion of the
    local response is returned (text before the [ESCALATE] marker).
    """
    should_escalate, forwarded_query = detect_escalation(response)

    if not should_escalate:
        return ChatResult(text=response, probed_keys=probed_keys or [])

    sanitized, _ = sanitize_for_cloud(forwarded_query, mode=config.sanitization.mode)
    if sanitized is not None:
        logger.info("[ESCALATE] → HA (source=%s): %s", match_source, sanitized[:100])
        result = _escalate_to_ha_agent(sanitized, ha_client, config, language=language)
        if result is not None:
            return result
        if sota_agent is not None:
            logger.info("[ESCALATE] → SOTA fallback (source=%s): %s", match_source, sanitized[:100])
            return _escalate_to_sota(
                sanitized,
                sota_agent,
                config,
                speaker=speaker,
                history=history,
                language=language,
            )

    # All escalation paths exhausted — return pre-escalation text from local model
    local_text = response.split("[ESCALATE]")[0].strip()
    return ChatResult(text=local_text or "I'm not sure about that.", probed_keys=probed_keys or [])


def _load_simhash_registry(registry_path) -> dict:
    """Load SimHash values from registry for probe verification."""
    registry = {}
    if registry_path.exists():
        from paramem.backup.encryption import read_maybe_encrypted

        raw = json.loads(read_maybe_encrypted(registry_path).decode("utf-8"))
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
    """Build chat messages enforcing strict user/assistant alternation.

    Mistral requires: system → user → assistant → user → ...
    HA may send non-alternating history, so we enforce the pattern here.
    """
    pairs = []
    if history:
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role", "user")
            content = turn.get("text", "")
            if role in ("user", "assistant") and content:
                pairs.append({"role": role, "content": content})

    merged = []
    for msg in pairs:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    while merged and merged[0]["role"] == "assistant":
        merged.pop(0)

    messages = [{"role": "system", "content": system_prompt}] + merged

    if messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n" + text
    else:
        messages.append({"role": "user", "content": text})

    return adapt_messages(messages, tokenizer)
