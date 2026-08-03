"""Chat inference — intent-driven routing dispatch.

Dispatch is on ``RoutingPlan.intent`` populated by the router:

1. ``PERSONAL`` → local adapter probe + base-model reasoning
   (``_probe_and_reason``); if the local model emits ``[ESCALATE]``,
   the forwarded query flows through HA → cloud per
   :func:`_handle_escalation`.
2. ``COMMAND`` → HA conversation agent first (verbatim sanitized
   query), cloud fallback only when HA is unreachable.
3. ``GENERAL`` → HA first, cloud fallback.
4. ``UNKNOWN`` (intent could not be established — no classifier
   config, no encoder/exemplars loaded, below-margin confidence, or an
   unparseable LLM verdict) is **not** positively PERSONAL: it grants
   no personal-memory access and does not block cloud escalation —
   routed identically to ``GENERAL`` (HA first, cloud fallback).  Only
   a positive PERSONAL verdict reaches the local parametric-memory
   probe branch.

Speaker scoping (``RoutingPlan.steps``) is the privacy boundary — only
the resolved speaker's keys can populate ``keys_to_probe``.

There is ONE personal verdict, computed once in :func:`handle_chat`: the
union of the intent classifier's ``PERSONAL`` verdict and
:func:`~paramem.server.sanitizer.is_self_referential`'s verdict on the
raw text.  It travels the call tree as ``is_personal`` and gates both the
CLOUD leg and the choice of the local parametric-memory probe branch —
HA is local and stays reachable as a tool fallback on every path.
The one exception is the model-authored forwarded query behind
``[ESCALATE]``: it is a different artifact from the turn, so
:func:`_maybe_escalate` computes a second verdict on it with the same
predicate and suppresses BOTH hops (``ha_agent_id`` is operator-pointed
and may be cloud-backed) when that verdict is personal.
:func:`answer_via_cloud` is the sole cloud-egress funnel, for both local
mode and cloud-only mode (``paramem.server.app._relay_route`` and the
``/chat`` forced-route cloud-only branch route through it too, with
``model``/``tokenizer`` left ``None`` — cloud-only runs with no local model
and therefore no ParaMem-held knowledge to protect).
:func:`_escalate_to_cloud` is the transport primitive underneath it and has
exactly one caller, :func:`answer_via_cloud`.

Fallback chain at every escalation point: HA → cloud → local base model
(``_base_model_answer``).  ``_escalate_to_ha_agent`` is HA-only;
callers own the cloud fallback.
"""

import logging
from dataclasses import dataclass, field

from paramem.cloud.providers.base import CloudAgent
from paramem.evaluation.recall import generate_answer
from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX
from paramem.models.loader import adapt_messages, base_model_inference, grad_checkpointing_disabled
from paramem.server.config import ServerConfig
from paramem.server.escalation import detect_escalation
from paramem.server.router import Intent, RoutingPlan
from paramem.server.sanitizer import is_self_referential
from paramem.server.tools.ha_client import HAClient
from paramem.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


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


def _build_speaker_prefix(
    speaker_id: str | None,
    language: str | None,
    config: ServerConfig | None,
) -> str:
    """Assemble the speaker + language prefix for the LOCAL reasoning prompt.

    The "You are speaking with X" line is fed the raw ``speaker{N}`` token,
    never the display name — identity stays in token space in every
    model-facing surface (recalled facts, reasoning context, generated
    replies).  A human-readable name is substituted only at the reply
    boundary, by :func:`~paramem.server.speaker.resolve_speaker_tokens`,
    after the turn is generated and persisted.

    This function is called ONLY on the local reasoning leg (from
    :func:`_probe_and_reason` and :func:`_base_model_answer`).  The cloud leg
    (:func:`_escalate_to_cloud`) never calls it and carries no identity line
    at all — cloud never learns the speaker's id or name.

    Args:
        speaker_id: The speaker's canonical ``speaker{N}`` token, or ``None``
            when unresolved.  Anonymous/undisclosed speakers are included —
            their raw token is the identity line's payload the same as a
            named speaker's; there is no name-presence gate here (see
            :func:`_build_system_prompt`).
        language: BCP-47 language code, or ``None`` / ``"en"`` when no
            instruction is needed.
        config: Server config, used to derive the language display name via
            ``config.tts.language_name``.

    Returns:
        A prefix string (possibly empty) ready to be prepended to the base
        system prompt.
    """
    parts: list[str] = []
    if speaker_id:
        parts.append(f"You are speaking with {speaker_id}.")
    lang_instr = _language_instruction(language, config)
    if lang_instr:
        parts.append(lang_instr)
    return " ".join(parts)


def _build_system_prompt(
    speaker_id: str | None,
    language: str | None,
    config: ServerConfig,
) -> str:
    """Assemble the complete system prompt for a LOCAL reasoning generate call.

    THE single assembly point for the identity + language prefix on top of
    ``config.voice.load_prompt()``.  Both :func:`_probe_and_reason` and
    :func:`_base_model_answer` call this — collapsing what were previously
    two byte-identical four-line blocks — so the two legs cannot drift from
    each other.

    Identity for the LOCAL reasoning prompt is the raw ``speaker{N}`` token,
    not the display name (see :func:`_build_speaker_prefix`).  The prefix is
    present iff ``speaker_id`` is resolved — anonymous/undisclosed speakers
    included; there is no display-name gate here.  A human-readable name is
    substituted only at the reply boundary
    (:func:`~paramem.server.speaker.resolve_speaker_tokens`), never in a
    model-facing prompt.

    Args:
        speaker_id: The speaker's canonical ``speaker{N}`` token, or ``None``
            when unresolved — gates whether the identity line is included
            at all.
        language: BCP-47 language code, or ``None``/``"en"`` for no
            instruction.
        config: Server config — supplies the language display name and the
            base voice prompt (``config.voice.load_prompt()``).

    Returns:
        The complete system prompt: the identity/language prefix (when any)
        followed by the base voice prompt.
    """
    prefix = _build_speaker_prefix(speaker_id, language, config)
    base_prompt = config.voice.load_prompt()
    return f"{prefix} {base_prompt}" if prefix else base_prompt


@dataclass
class ChatResult:
    text: str
    escalated: bool = False
    probed_keys: list[str] = field(default_factory=list)


def _is_personal_interrogative(text: str, config: ServerConfig, *, is_personal: bool) -> bool:
    """Return True when *text* is both personal-class and interrogative.

    THE one implementation of this conjunction — two independent gates
    call it rather than re-deriving ``is_personal and _is_interrogative(...)``
    themselves:

    * :func:`_abstain_if_applicable` — additionally gates on
      ``config.abstention.enabled`` (a feature toggle).
    * ``paramem.server.app._relay_route``'s no-identity short-circuit —
      deliberately NOT gated on ``config.abstention.enabled``.  A
      speakerless caller has no store to abstain *from*; refusing the
      personal interrogative there is a structural impossibility (there is
      no identity for the question to be about), not a feature the
      operator can toggle off.

    Args:
        text: The turn to classify.
        config: Server config — supplies ``config.sentence_type`` for the
            interrogative classifier.
        is_personal: Precomputed personal-class verdict.  Callers compute
            this independently — ``handle_chat``'s union of the intent
            classifier and :func:`~paramem.server.sanitizer.is_self_referential`
            for an identified speaker; plain ``is_self_referential`` for a
            speakerless relay turn, which has no intent classifier to
            consult (routing requires a resolved ``speaker_id``).

    Returns:
        ``True`` iff *is_personal* and *text* is interrogative.
    """
    from paramem.server.router import _is_interrogative

    return is_personal and _is_interrogative(text, config=config.sentence_type)


def _abstain_if_applicable(
    text: str,
    config: ServerConfig,
    *,
    is_personal: bool,
    speaker_id: str | None = None,
    router=None,
) -> tuple[ChatResult, str] | None:
    """Decide whether to short-circuit with the canned abstention response.

    Gate: ``config.abstention.enabled`` AND :func:`_is_personal_interrogative`
    (``is_personal`` AND the query is interrogative per
    :func:`paramem.server.router._is_interrogative`).  When the gate fires,
    returns ``(canned_chat_result, exit_via_label)``; otherwise returns
    ``None`` and the caller continues the escalation chain.

    The cold-start variant fires when ``speaker_id`` is set but the
    router has no keys for them yet (between enrollment and the first
    consolidation cycle).  The standard ``response`` covers the
    coverage-gap case (speaker has facts but this query missed them).
    Callers that don't know the cold-start state — e.g. a callee deep
    in the dispatch tree where probes already succeeded — can omit
    ``router`` / ``speaker_id`` and the helper defaults to the canned
    response.

    The label distinguishes ``"abstention_cold_start"`` from
    ``"abstention_canned"`` for routing-diagnostics; callers update
    their own diag dicts from the returned label as needed.

    AbstentionBench (NeurIPS 2025) showed prompt-only abstention is
    unreliable at 7B-9B; this deterministic short-circuit is the only
    fix with zero hallucination risk on personal interrogatives that
    parametric memory cannot answer.
    """
    if not (
        config.abstention.enabled
        and _is_personal_interrogative(text, config, is_personal=is_personal)
    ):
        return None
    is_cold_start = bool(speaker_id) and (
        router is None or not router._speaker_key_index.get(speaker_id)
    )
    response_text = (
        config.abstention.load_cold_start_response()
        if is_cold_start
        else config.abstention.load_response()
    )
    label = "abstention_cold_start" if is_cold_start else "abstention_canned"
    return ChatResult(text=response_text), label


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
    speaker_id: str | None = None,
    language: str | None = None,
    effective_mode: str | None = None,
    memory_store=None,
) -> ChatResult:
    """Process a chat message via intent-keyed dispatch.

    Routing reads ``RoutingPlan.intent`` populated by the router's
    classify_intent() pass:

    * ``PERSONAL`` → local PA probe + reason.  HA is reachable from the
      local model via ``[ESCALATE]`` and from the no-layers branch as a
      tool fallback.  **cloud is never reached** — personal-class queries
      stay off the cloud (privacy invariant, threaded as ``is_personal``
      through the call tree).
    * ``COMMAND`` / ``GENERAL`` / ``UNKNOWN`` → HA first (tools, live
      state), cloud fallback (reasoning).  ``UNKNOWN`` (intent could not
      be established) is not positively PERSONAL, so it is treated
      identically to ``GENERAL`` — no personal-memory access, and cloud
      escalation stays available.

    ``is_personal`` is computed ONCE here, as the union of that intent
    verdict and :func:`~paramem.server.sanitizer.is_self_referential`'s
    verdict over ``text``.  A ``COMMAND``/``GENERAL`` turn that refers to
    the speaker is therefore personal too, even though the classifier
    said otherwise.  The verdict gates the CLOUD leg and selects the
    local parametric-memory probe branch; HA stays reachable on every
    path, and a personal turn that neither HA nor the cloud answered
    falls to abstention before the base model.

    The ``is_residual`` diagnostic tracks "did any graph signal fire?"
    for the routing-quality metric independent of the intent decision —
    ``True`` when neither PA steps nor HA domains were produced.

    When ``config.debug`` is True a per-request routing-decision
    diagnostic is emitted via ``logging.info(extra={"routing": …})`` at
    function exit.

    Raises:
        ValueError: if ``speaker_id`` is ``None``.  ``handle_chat`` requires
            a resolved speaker — speakerless requests are served entirely by
            the relay path (``paramem.server.app._relay_route``, forked at
            the ``ServingPath`` boundary before ``handle_chat`` is ever
            called) and never reach here.
    """
    if speaker_id is None:
        raise ValueError(
            "handle_chat requires a resolved speaker_id — speakerless requests "
            "are served by the relay path (_relay_route), never by handle_chat."
        )
    routing_diags: dict = {
        "conversation_id": conversation_id,
        "intent": Intent.UNKNOWN.value,
        "paths_attempted": [],
        "fallthrough_reason": None,
        "exit_via": None,
        "is_residual": False,
    }
    try:
        with grad_checkpointing_disabled(model):
            plan = None

            # Dual-graph entity routing.  The temporal-query branch (filter
            # keys by date range) was retired — its data source (combined
            # registry with last_seen_at / status fields) was never populated
            # by production paths, so the filter always returned an empty
            # list and the branch was inert.  If we re-introduce temporal
            # queries, the writer side needs to be designed first.
            if router is not None:
                plan = router.route(text, speaker_id=speaker_id)
            if plan is not None:
                routing_diags["intent"] = plan.intent.value

            intent = plan.intent if plan is not None else Intent.UNKNOWN
            # First arm of the personal verdict.  Only a positive PERSONAL
            # verdict grants memory access and blocks cloud escalation.
            # Every other intent — including UNKNOWN (no IntentConfig,
            # classifier unavailable, below-margin confidence) — routes
            # through the normal HA → cloud → base-model chain, same as
            # GENERAL.
            intent_is_personal = intent == Intent.PERSONAL

            # THE personal verdict, computed once.  Two arms, unioned: the
            # intent classifier and the self-reference gate.  Everything
            # downstream reads ``is_personal``; neither arm is re-derived
            # anywhere else.
            is_self_ref = is_self_referential(
                text,
                personal_referent_config=config.personal_referent,
            )
            is_personal = intent_is_personal or is_self_ref
            routing_diags["is_self_referential"] = is_self_ref

            # The former anonymous deny-by-default branch is gone: a
            # speakerless caller never reaches ``handle_chat`` (the
            # ``ServingPath`` boundary in ``paramem.server.app`` forks
            # speakerless requests to the relay path before this function is
            # called — see the contract assertion above), so ``speaker_id``
            # is always resolved here and there is nothing left to deny.

            # PERSONAL → local PA probe + reason.  No cloud anywhere on this
            # path: is_personal=True suppresses every internal _escalate_to_cloud
            # call (no-layers branch, post-reason [ESCALATE], base-model
            # fallthrough).  HA stays reachable as a tool fallback.
            if is_personal and plan is not None and plan.steps:
                routing_diags["paths_attempted"].append("personal")
                routing_diags["exit_via"] = "personal_probe"
                return _probe_and_reason(
                    text,
                    plan,
                    history,
                    model,
                    tokenizer,
                    config,
                    cloud_agent=cloud_agent,
                    ha_client=ha_client,
                    speaker=speaker,
                    speaker_id=speaker_id,
                    language=language,
                    effective_mode=effective_mode,
                    is_personal=True,
                    memory_store=memory_store,
                )

            # COMMAND / GENERAL / UNKNOWN (and the defensive PERSONAL-without-
            # steps path) → HA first, cloud fallback.  is_personal still gates
            # cloud so a defensive PERSONAL request never reaches the cloud.
            intent_label = intent.value
            routing_diags["paths_attempted"].append(intent_label)
            logger.info("Intent dispatch: %s → HA first", intent_label)
            result = _escalate_to_ha_agent(text, ha_client, config, language=language)
            if result is not None:
                routing_diags["exit_via"] = f"{intent_label}_ha"
                return result
            cloud_result = answer_via_cloud(
                text,
                cloud_agent,
                config,
                is_personal=is_personal,
                model=model,
                tokenizer=tokenizer,
                speaker=speaker,
                speaker_id=speaker_id,
                history=history,
                language=language,
            )
            if cloud_result is not None:
                routing_diags["exit_via"] = f"{intent_label}_cloud"
                logger.info("HA failed, routing to cloud agent")
                return cloud_result
            if is_personal:
                routing_diags["fallthrough_reason"] = "personal_cloud_blocked"

            # Abstention: personal interrogative with no local match → canned response.
            # The bare base model would otherwise confabulate personal data here
            # (e.g. "Where do I live?" → "New York City" on an untrained adapter).
            # Declarative personal turns (introductions, fact-sharing) are not a
            # confabulation risk — the user is the source of the facts in the same
            # turn — so they fall through to the base model for conversational
            # acknowledgement.  The interrogative gate (inside the helper)
            # distinguishes the two, and ``is_personal`` gates the whole thing.
            #
            # Reached only after HA and cloud both produced nothing.  The
            # companion call site inside ``_probe_and_reason`` covers the
            # parallel case where probes ran but recalled nothing.
            abstention = _abstain_if_applicable(
                text,
                config,
                is_personal=is_personal,
                speaker_id=speaker_id,
                router=router,
            )
            if abstention is not None:
                result, label = abstention
                routing_diags["paths_attempted"].append("abstention")
                routing_diags["exit_via"] = label
                logger.info(
                    "Abstention: self-referential query + no local match (%s)",
                    label,
                )
                return result

            # All cloud services failed — local base model as last resort
            routing_diags["paths_attempted"].append("base")
            routing_diags["exit_via"] = "base_model"
            return _base_model_answer(
                text,
                history,
                model,
                tokenizer,
                config,
                cloud_agent=cloud_agent,
                ha_client=ha_client,
                speaker=speaker,
                speaker_id=speaker_id,
                language=language,
                is_personal=is_personal,
            )
    finally:
        if getattr(config, "debug", False):
            # is_residual: the routing plan landed with no probe steps and
            # no HA domains — i.e. no deterministic signal fired and the
            # classifier residual (encoder cosine or LLM generate) drove the
            # intent verdict.  Tracks whether the routing-quality metric
            # should count this query toward the residual classifier's
            # evaluation.
            routing_diags["is_residual"] = bool(
                plan is not None and not plan.steps and not plan.ha_domains
            )
            logger.info("routing decision", extra={"routing": routing_diags})


def _escalate_to_ha_agent(
    text: str,
    ha_client: HAClient | None,
    config: ServerConfig,
    language: str | None = None,
) -> ChatResult | None:
    """Forward to the HA conversation agent.

    Returns None if HA is unavailable or the request fails. Callers own
    the cloud fallback — this function is HA-only.
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


CLOUD_PROMPT = (
    "You are continuing a conversation as a personal assistant. "
    "Derive your persona, tone, and conversational style from the "
    "preceding conversation. Answer clearly and concisely in 1-3 spoken "
    "sentences. Do not use markdown, lists, or structured formatting."
)


def _sanitize_history(history: list[dict] | None) -> list[dict]:
    """Drop-gate conversation history for cloud: self-referential turns are removed.

    Unconditional — there is no pass-through or warn-only setting.  A
    history turn that :func:`~paramem.server.sanitizer.is_self_referential`
    flags never egresses, whether or not the current turn is being
    placeholdered for privacy.

    Args:
        history: Conversation turns to gate.  Only the last
            :data:`MAX_HISTORY_TURNS` are considered; empty-text turns are
            dropped.

    Returns:
        The surviving turns as ``{"role", "text"}`` dicts, in order.
    """
    if not history:
        return []
    sanitized = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        if not text:
            continue
        if is_self_referential(text):
            logger.info("Dropped self-referential history turn from cloud payload")
            continue
        sanitized.append({"role": role, "text": text})
    return sanitized


def answer_via_cloud(
    text: str,
    cloud_agent: CloudAgent | None,
    config: ServerConfig,
    *,
    is_personal: bool = False,
    model=None,
    tokenizer=None,
    speaker: str | None = None,
    speaker_id: str | None = None,
    history: list[dict] | None = None,
    language: str | None = None,
    cloud_permitted: bool = True,
) -> ChatResult | None:
    """Apply the configured cloud-egress policy and call cloud accordingly.

    The sole cloud-egress funnel: every caller — local-mode routing (this
    module) and cloud-only routing
    (``paramem.server.app._relay_route`` and the ``/chat`` forced-route
    cloud-only branch) — reaches :func:`_escalate_to_cloud` only through
    here.

    Returns the cloud result on success, or ``None`` when policy or per-query
    safety blocks the call (caller falls through to the next mechanism in the
    escalation chain — typically the base model, HA's own fallback, or a
    canned limited-mode response).

    The policy branches on one axis — can this call anonymize the outbound
    text?  ``model``/``tokenizer`` decide it:

    * **Can anonymize** (``model`` and ``tokenizer`` both given — always true
      on the local-mode path): the ``config.sanitization.cloud_mode`` policy
      below applies.

      +-------------+----------------------+----------------------+
      | mode        | PERSONAL query       | non-PERSONAL query   |
      +=============+======================+======================+
      | ``block``   | None (blocked)       | cloud verbatim        |
      | ``anonymize`` | anon → cloud → deanon | anon → cloud → deanon |
      | ``both``    | None (blocked)       | anon → cloud → deanon |
      +-------------+----------------------+----------------------+

      Per-query safety: when an anonymizing path is selected and the local
      anonymizer fails to produce a mapping (model/extraction failure, parse
      failure), this call returns ``None`` so the caller falls back without
      sending anything to the cloud.  The config knob is unchanged for the
      next query.

    * **Cannot anonymize** (``model`` or ``tokenizer`` is ``None`` — this is
      cloud-only mode: no local model, therefore no ParaMem-held knowledge to
      protect).  ``cloud_mode`` and ``is_personal`` do not apply — there is
      no store for ``is_personal`` to be about.  The current turn egresses
      **verbatim** iff ``cloud_permitted``; history is still drop-gated via
      :func:`_sanitize_history`.  Returns ``None`` when ``cloud_permitted``
      is ``False``.

    ``cloud_permitted`` defaults to ``True``: on the local-mode path, cloud
    egress is already gated upstream by ``cloud_agent`` presence (a disabled
    cloud agent makes ``cloud_agent`` ``None``, handled below), so local
    callers never compute it and pass the default.  Cloud-only callers
    compute it from ``_state["cloud_only_reason"]`` and
    ``config.cloud.allow_degraded_serving`` (whether the cloud-only state is
    voluntary) and thread the result in explicitly.
    """
    if cloud_agent is None:
        return None

    if model is None or tokenizer is None:
        # Cannot anonymize — cloud-only mode, no ParaMem-held knowledge to
        # protect.  Verbatim egress iff cloud_permitted; history is still
        # drop-gated — an old turn can be personal even in cloud-only mode.
        if not cloud_permitted:
            return None
        sanitized_history = _sanitize_history(history)
        return _escalate_to_cloud(
            text,
            cloud_agent,
            config,
            sanitized_history=sanitized_history,
            language=language,
        )

    cloud_mode = config.sanitization.cloud_mode
    if cloud_mode not in {"block", "anonymize", "both"}:
        # Unknown / mock value — fall back to the safest mode (block).
        # Production paths can't reach this branch because
        # SanitizationConfig.__post_init__ validates the field; this guard
        # protects test mocks and any future config drift.
        cloud_mode = "block"

    blocks_personal = cloud_mode in {"block", "both"}
    anonymizes_outbound = cloud_mode in {"anonymize", "both"}

    if is_personal and blocks_personal:
        return None

    if anonymizes_outbound:
        from paramem.cloud.deanonymize import CloudScope, deanonymize_text
        from paramem.cloud.placeholders import _substitute_whole_words
        from paramem.graph.flows import anonymize_turn

        # Anonymize ONLY the current-turn text — the model-facing
        # anonymized transcript comes back on ``payload.anon_transcript``.
        payload = anonymize_turn(
            text,
            model,
            tokenizer,
            speaker_id=speaker_id,
            speaker_name=speaker,
            scrub=set(config.sanitization.scrub),
        )
        if payload.status == "failed":
            # Per-query block: extraction error, anonymizer parse failure or
            # a missing/empty model-authored transcript (fail-closed), or
            # an empty model-authored transcript after the marker strip.
            # Privacy-safe — cloud call is suppressed.  Distinct from
            # ``status == "opted_out"``, which proceeds with the verbatim
            # transcript below.
            return None

        # History: under an anonymizing cloud_mode, history is
        # NOT bundled into a single anonymized transcript with the current
        # turn (that would show the cloud the history twice and forces
        # multi-turn text reproduction on the local 7B model, which it
        # doesn't do reliably).  Instead: (i) drop-gate each turn via
        # ``_sanitize_history`` — content-only, no ``speaker_id`` (see
        # ``is_self_referential``'s docstring: it takes no speaker_id
        # parameter) — then (ii) substitute through ``payload.forward``
        # — no second LLM call.  Accepted residual: ``payload.forward``
        # only covers entities the anonymizer named in the CURRENT turn,
        # so a personal entity appearing ONLY in history is dropped by the
        # gate but not placeholdered.  See benchmarking.md.
        drop_gated_history = _sanitize_history(history)
        sanitized_history = [
            {**turn, "text": _substitute_whole_words(turn["text"], payload.forward)}
            for turn in drop_gated_history
        ]

        result = _escalate_to_cloud(
            payload.anon_transcript,
            cloud_agent,
            config,
            sanitized_history=sanitized_history,
            language=language,
        )
        # ``sent`` is the current-turn text only at this step — history
        # turns are anonymized via ``payload.forward`` above, not via a
        # cloud round trip, so they carry no NEW placeholder tokens to
        # scope; a scoped-observed deanon of the cloud's response.
        scope = CloudScope.response(payload, cloud_bindings=None, sent=(payload.anon_transcript,))
        deanon_text = deanonymize_text(scope, result.text)
        if deanon_text is None:
            # Fail-closed: a declared-but-unobserved placeholder (or
            # otherwise unresolved token) survived in the cloud's
            # response — never forward it with a residual placeholder.
            logger.warning("Cloud response carried an unresolved placeholder — blocking")
            return None
        result.text = deanon_text
        return result

    # cloud_mode=block + non-PERSONAL: current turn goes verbatim (the
    # personal verdict already cleared it).  History is still drop-gated —
    # an old turn can be personal even when this one is not.
    sanitized_history = _sanitize_history(history)
    return _escalate_to_cloud(
        text,
        cloud_agent,
        config,
        sanitized_history=sanitized_history,
        language=language,
    )


def _escalate_to_cloud(
    text: str,
    cloud_agent: CloudAgent,
    config: ServerConfig,
    sanitized_history: list[dict] | None = None,
    language: str | None = None,
) -> ChatResult:
    """Route to cloud model for reasoning-heavy queries.

    Passes conversation history so the cloud model can derive persona,
    tone, and style from the conversation context.

    This is the transport primitive, not a funnel: it makes no policy
    decision of its own and has exactly one caller,
    :func:`answer_via_cloud`, which is the sole cloud-egress funnel for
    both local mode and cloud-only mode.

    The system prompt carries NO identity line at all — no ``speaker_id``
    token and no display name.  Unlike the local reasoning leg
    (:func:`_build_speaker_prefix`, fed the ``speaker{N}`` token), the cloud
    system prompt is the bare :data:`CLOUD_PROMPT` plus only the language
    instruction.  This is scoped to the system prompt only: sanitized
    history (:func:`_sanitize_history`) can still carry a ``speaker{N}``
    token verbatim through ``answer_via_cloud``'s history-substitution path —
    an accepted posture, not a leak — so "cloud never learns who it is
    talking to" would overclaim.

    Args:
        text: The query text.  Sanitized/anonymized (or, in cloud-only mode,
            deliberately left verbatim per policy) by :func:`answer_via_cloud`
            before this call — that is the one place the egress policy is
            decided.
        cloud_agent: Cloud agent to delegate to.
        config: Server config.
        sanitized_history: Conversation history turns, ALREADY drop-gated
            (and, under an anonymizing ``cloud_mode``, placeholdered) by
            :func:`answer_via_cloud` — this function does not sanitize.
        language: BCP-47 language code.
    """
    sanitized_history = sanitized_history or []

    lang_instr = _language_instruction(language, config)
    prompt = (lang_instr + " " + CLOUD_PROMPT) if lang_instr else CLOUD_PROMPT

    logger.info(
        "cloud escalation (%d history turns): %s",
        len(sanitized_history),
        text[:100],
    )
    response = cloud_agent.call(
        query=text,
        system_prompt=prompt,
        history=sanitized_history,
    )
    if response.text:
        return ChatResult(text=response.text, escalated=True)
    return ChatResult(text="I couldn't get an answer right now.", escalated=True)


# A decoded reply re-tokenized with the same tokenizer is not guaranteed to
# re-encode to exactly the token count that was generated: a hard
# max_new_tokens cut can land mid-subword, and the decode->encode round trip
# through a BPE tokenizer is not exactly invertible at that boundary. A
# reply within this many tokens of the cap is still treated as a cap-hit —
# without the tolerance, an off-by-one/two round-trip drift would silently
# under-detect truncation and leave a mid-sentence cut untrimmed.
_CAP_HIT_TOKEN_TOLERANCE = 2


def _generate_local_reply(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    *,
    speaker_id: str | None,
    language: str | None,
) -> tuple[str, bool]:
    """THE local reasoning generate — the one place a served reply is produced.

    System-prompt assembly, message shaping, chat-template rendering and the
    ``generate_answer`` call live here exactly once; :func:`_probe_and_reason`
    and :func:`_base_model_answer` both call this instead of carrying their
    own byte-identical tail.  ``max_new_tokens`` comes from
    ``config.inference.max_response_tokens`` — the single literal-free site
    for that cap.

    Runs inside :func:`~paramem.models.loader.base_model_inference`, which
    disables gradient checkpointing (so the KV cache stays live) and, when
    *model* is a ``PeftModel``, disables the active adapter for the
    duration — reasoning always runs on the base weights, whichever adapter
    was last active for probing.

    Cap-hit detection lives HERE — this is the one place the fact is
    derivable, since only the generate call knows the actual token budget it
    ran against. The decoded reply is re-tokenized with *tokenizer* and
    compared against ``max_new_tokens`` within
    :data:`_CAP_HIT_TOKEN_TOLERANCE`; the boolean is threaded to
    :func:`_maybe_escalate` so trimming applies ONLY to a reply the cap
    actually cut — a complete reply is never mutated.

    Args:
        text: The (possibly context-augmented) query text.
        history: Prior turns for :func:`_build_messages`, or ``None``.
        model: The (optionally PEFT-wrapped) model to generate from.
        tokenizer: The model's tokenizer.
        config: Server config — supplies the system prompt and the response
            token cap.
        speaker_id: The speaker's canonical ``speaker{N}`` token, or ``None``.
        language: BCP-47 language code, or ``None``/``"en"``.

    Returns:
        ``(reply_text, is_truncated)`` — *reply_text* is untrimmed (trimming
        happens in :func:`_maybe_escalate`, not here) and *is_truncated* is
        True iff the reply's re-tokenized length is within
        :data:`_CAP_HIT_TOKEN_TOLERANCE` of ``max_new_tokens``.
    """
    system_prompt = _build_system_prompt(speaker_id, language, config)
    messages = _build_messages(text, history, system_prompt, tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    max_new_tokens = config.inference.max_response_tokens
    with base_model_inference(model):
        reply = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
    reply_tokens = estimate_tokens(reply, tokenizer)
    is_truncated = reply_tokens >= max_new_tokens - _CAP_HIT_TOKEN_TOLERANCE
    return reply, is_truncated


def _probe_and_reason(
    text: str,
    plan: RoutingPlan,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    is_personal: bool = False,
    effective_mode: str | None = None,
    memory_store=None,
) -> ChatResult:
    """Probe adapters in memory hierarchy order, assemble layered context.

    Builds a ``keys_by_adapter`` dict from ``plan.steps`` (preserving router
    order: procedural → episodic → semantic → session adapters newest-first),
    dispatches to ``MemoryStore.probe`` for cache resolution + on-miss source
    delegation, then reassembles per-layer facts for context augmentation.

    Cache hits return in O(1).  On cache miss the :class:`MemorySource` built
    by :func:`~paramem.memory.source.build_memory_source` resolves the entry and
    the result is memoized back into the cache when
    ``config.inference.preload_cache`` is True.

    After probing, restores the model to the ``episodic`` adapter so the next
    query starts from a predictable state. The reasoning phase uses
    ``model.disable_adapter()`` so the active adapter during generation does
    not matter — only the post-return state (restored here) does.

    Privacy gate: ``is_personal`` flows through to every internal cloud
    fallback site (no-layers branch, base-model fallthrough, post-reason
    [ESCALATE]).  Personal-class queries never reach the cloud.

    Each result dict carries a ``fact_text`` field pre-rendered for the
    bullet list.
    """
    from paramem.memory.source import build_memory_source
    from paramem.models.loader import switch_adapter

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

    # Mode-aware on-miss source from the one factory.  The MemoryStore cache is
    # RAM-only and is the fast path; the source is the slow-path fallback when a
    # key isn't already cached.  ``None`` (train mode, no model) leaves the probe
    # cache-only.
    _active_mode = effective_mode if effective_mode else config.consolidation.mode
    source = build_memory_source(
        mode=_active_mode,
        adapter_dir=config.adapter_dir,
        batch_size=config.consolidation.recall_probe_batch_size,
        model=model,
        tokenizer=tokenizer,
    )

    probe_results = memory_store.probe(
        keys_by_adapter,
        source=source,
        speaker_id=speaker_id,
        memoize=config.inference.preload_cache,
    )

    # Restore predictable adapter state after weight probing: episodic is
    # the main adapter for PM inference.  The reasoning phase uses
    # disable_adapter() so the active adapter during generation does not
    # matter — only the post-return state (restored here) does.  No-op in
    # simulate mode where probing didn't touch the model.
    if (
        _active_mode != "simulate"
        and model is not None
        and hasattr(model, "peft_config")
        and "episodic" in model.peft_config
    ):
        switch_adapter(model, "episodic")

    # Reassemble per-step facts so each adapter's results go to its layer.
    layers: dict[str, list[str]] = {}
    successful_keys = []

    for step in plan.steps:
        layer_facts = []
        for key in step.keys_to_probe:
            result = probe_results.get(key)
            if result and "failure_reason" not in result:
                # fact_text is guaranteed on every success result from
                # probe_keys_grouped_by_adapter / probe_keys_from_graph.
                # The get() fallback covers mocked/legacy callers that return
                # a bare {answer: ...} dict without the field.
                layer_facts.append(f"- {result.get('fact_text', result.get('answer', ''))}")
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
            "All %d probed key(s) failed, escalating via HA%s (intent=%s)",
            sum(len(s.keys_to_probe) for s in plan.steps),
            "" if is_personal else " → cloud",
            plan.intent.value,
        )
        result = _escalate_to_ha_agent(text, ha_client, config, language=language)
        if result is not None:
            return result
        cloud_result = answer_via_cloud(
            text,
            cloud_agent,
            config,
            is_personal=is_personal,
            model=model,
            tokenizer=tokenizer,
            speaker=speaker,
            speaker_id=speaker_id,
            history=history,
            language=language,
        )
        if cloud_result is not None:
            return cloud_result
        # Abstention: ``_probe_and_reason`` is reached only for PERSONAL with
        # non-empty plan.steps (handle_chat dispatch).  Probes failed and HA
        # had no tool answer either; the base model has no context here
        # (``not layers`` means no facts were recalled), so generating an
        # answer would be unconditional confabulation.
        #
        # ``router`` and ``speaker_id`` are not threaded into this function —
        # default of ``None`` makes :func:`_abstain_if_applicable` return the
        # canned response (cold-start can't apply: reaching here means the
        # router built probes from the speaker's existing keys, so the
        # speaker has facts and the coverage-gap response fits).
        abstention = _abstain_if_applicable(text, config, is_personal=is_personal)
        if abstention is not None:
            result, _label = abstention
            logger.info(
                "Abstention: PA-empty personal interrogative in _probe_and_reason "
                "(probes=%d failed, HA returned None)",
                sum(len(s.keys_to_probe) for s in plan.steps),
            )
            return result

        return _base_model_answer(
            text,
            history,
            model,
            tokenizer,
            config,
            cloud_agent=cloud_agent,
            ha_client=ha_client,
            speaker=speaker,
            speaker_id=speaker_id,
            language=language,
            is_personal=is_personal,
        )

    total_facts = sum(len(f) for f in layers.values())
    logger.info("Total recalled: %d facts from %d layers", total_facts, len(layers))

    # Assemble layered context: procedural → episodic (incl. interim slots) → semantic.
    # Later sections sit closer to the query, giving them higher recency bias.
    #
    # Adapter-name mapping: probe results land in ``layers`` under the
    # ``step.adapter_name`` used during routing.  For interim windows that name
    # is ``"episodic_interim_<stamp>"`` (per router.reload's
    # do-not-strip-stamps policy at router.py:274-283 — required so
    # ``switch_adapter`` lands on the trained slot).  The context-assembly
    # layer is conceptually still "episodic", so we collect every
    # ``episodic*`` adapter's facts under the single ``Recent knowledge``
    # bucket.  Multiple interim slots are emitted newest-stamp-first,
    # mirroring the router's probe-order policy.
    context_sections = []
    procedural_facts = layers.get("procedural")
    if procedural_facts:
        context_sections.append(f"[{LAYER_LABELS['procedural']}]\n" + "\n".join(procedural_facts))

    episodic_adapter_names = sorted(
        (n for n in layers if n == "episodic" or n.startswith(INTERIM_NAME_PREFIX)),
        key=lambda n: (n != "episodic", n),
        reverse=True,
    )
    episodic_facts: list[str] = []
    for adapter_name in episodic_adapter_names:
        episodic_facts.extend(layers[adapter_name])
    if episodic_facts:
        context_sections.append(f"[{LAYER_LABELS['episodic']}]\n" + "\n".join(episodic_facts))

    semantic_facts = layers.get("semantic")
    if semantic_facts:
        context_sections.append(f"[{LAYER_LABELS['semantic']}]\n" + "\n".join(semantic_facts))

    layered_context = "\n\n".join(context_sections)
    augmented_text = f"What you know about the speaker:\n\n{layered_context}\n\nQuestion: {text}"

    response, is_truncated = _generate_local_reply(
        augmented_text,
        history,
        model,
        tokenizer,
        config,
        speaker_id=speaker_id,
        language=language,
    )

    return _maybe_escalate(
        response,
        config,
        intent=plan.intent,
        probed_keys=successful_keys,
        cloud_agent=cloud_agent,
        ha_client=ha_client,
        speaker=speaker,
        speaker_id=speaker_id,
        history=history,
        language=language,
        is_personal=is_personal,
        model=model,
        tokenizer=tokenizer,
        is_truncated=is_truncated,
    )


def _base_model_answer(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    is_personal: bool = False,
) -> ChatResult:
    """Answer from base model without context — escalation candidate.

    ``is_personal`` propagates the privacy gate to ``_maybe_escalate`` so
    a base-model [ESCALATE] from a personal-class query cannot reach
    Cloud.
    """
    response, is_truncated = _generate_local_reply(
        text,
        history,
        model,
        tokenizer,
        config,
        speaker_id=speaker_id,
        language=language,
    )

    return _maybe_escalate(
        response,
        config,
        cloud_agent=cloud_agent,
        ha_client=ha_client,
        speaker=speaker,
        speaker_id=speaker_id,
        history=history,
        language=language,
        is_personal=is_personal,
        model=model,
        tokenizer=tokenizer,
        is_truncated=is_truncated,
    )


_SENTENCE_TERMINATORS = ".!?…"
# U+201C (LEFT DOUBLE QUOTATION MARK, "“") is included because German
# typography reuses that glyph as the CLOSING quote (opening is „, U+201E)
# — without it, a complete German reply ending in „…“ was misjudged as
# incomplete (the trailing-closer strip never reached the terminator
# underneath) and lost its closing quote to the (pre-truncation-gated) trim.
_TRAILING_CLOSERS = "\"'’”»)]}“"


def _is_boundary_terminator(text: str, index: int) -> bool:
    """Return True if ``text[index]`` is a sentence-terminating character.

    Excludes a ``.`` immediately preceded by a digit: a decimal (``3.5``)
    or an enumerated list marker (``3.``) is not a sentence boundary, even
    though ``.`` is a terminator in every other context. ``!``/``?``/``…``
    are never excluded — only ``.`` participates in this ambiguity.
    """
    ch = text[index]
    if ch not in _SENTENCE_TERMINATORS:
        return False
    if ch == "." and index > 0 and text[index - 1].isdigit():
        return False
    return True


def _trim_incomplete_sentence(text: str) -> str:
    """Drop a trailing incomplete sentence left by the token cap.

    Applied at exactly one site — :func:`_maybe_escalate`'s no-tag return —
    and ONLY when the caller has determined the reply was cap-truncated
    (:func:`_generate_local_reply`'s ``is_truncated``); a complete reply is
    never passed through this function at all, so it is never mutated. A
    closing quote/bracket/paren immediately after a terminator does not
    count as incompleteness (``He said "yes."`` / ``(yes.)`` both count as
    complete).

    No-op (returns *text* byte-identical) when:

    - *text* is empty or whitespace-only.
    - *text* already ends on a sentence terminator, ignoring any trailing
      closing punctuation.
    - *text* contains no terminator at all (nothing to trim back to).

    Otherwise returns *text* sliced through the LAST terminator found,
    plus any closing punctuation immediately following it. The slice
    always contains at least that terminator character (a non-whitespace
    character by construction), so trimming can never empty a reply — no
    separate guard is needed for that.

    Residual (accepted, not fixed here): an abbreviation period ("Dr.",
    "e.g.") is indistinguishable from a real sentence terminator by this
    function and may still cause a short-but-truthful cut on an
    already-truncated reply. This is a known residual on already-broken
    replies, not a promise that every cut lands on a true sentence
    boundary.

    Args:
        text: The generated reply text.

    Returns:
        *text*, or a prefix of it ending on a complete sentence.
    """
    if not text or not text.strip():
        return text

    trailing = text.rstrip().rstrip(_TRAILING_CLOSERS)
    if trailing and _is_boundary_terminator(trailing, len(trailing) - 1):
        return text

    last_idx = -1
    for i in range(len(text)):
        if _is_boundary_terminator(text, i):
            last_idx = i
    if last_idx == -1:
        return text

    end = last_idx + 1
    while end < len(text) and text[end] in _TRAILING_CLOSERS:
        end += 1

    return text[:end]


def _maybe_escalate(
    response: str,
    config: ServerConfig,
    intent: Intent | None = None,
    probed_keys: list[str] | None = None,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    speaker_id: str | None = None,
    history: list[dict] | None = None,
    language: str | None = None,
    is_personal: bool = False,
    model=None,
    tokenizer=None,
    is_truncated: bool = False,
) -> ChatResult:
    """Check for [ESCALATE] tag and route HA → cloud.

    This is the escalation-from-a-failed-local-answer path only.  Device
    control does not pass through here: an imperative with an HA entity
    match routes to HA directly in ``handle_chat`` and never reaches this
    function.

    HA agent has tools (search, device control, real-time data) so it
    gets first shot. Cloud handles queries that need pure reasoning.
    When both hops are suppressed or fail, the pre-escalation portion of
    the local response is returned (text before the [ESCALATE] marker).

    Privacy invariant: the forwarded query is a **model-authored** artifact,
    not the user's turn — on the personal path the model has already
    recalled facts from parametric memory and may have written them into
    the text after the tag.  It therefore carries its OWN verdict, computed
    here with the same :func:`~paramem.server.sanitizer.is_self_referential`
    predicate that produced the turn verdict.  A personal forwarded query
    suppresses the HA hop outright (``ha_agent_id`` is operator-configurable
    and is routinely pointed at a cloud-backed agent), and is unioned into
    the ``is_personal`` passed to :func:`answer_via_cloud` so the existing
    ``cloud_mode`` policy applies to the stronger of the two verdicts.

    ``model`` and ``tokenizer`` are forwarded to
    :func:`answer_via_cloud` so the anonymizer (when
    selected) can rewrite outbound text.

    ``is_truncated`` is :func:`_generate_local_reply`'s cap-hit verdict for
    *response* — the trim below runs ONLY when it is True, so a complete
    reply is never touched (kills the class of trim defects that mutated
    healthy replies, e.g. losing a trailing German closing quote). A
    response ending in a truncated tag fragment (``"…[ESCAL"``, not matched
    by ``detect_escalation``'s exact ``find``) is a byproduct of the SAME
    cap hit that set ``is_truncated``, so it is subsumed by the trim too —
    see ``handle_chat``/``_probe_and_reason`` for where the verdict
    actually originates (out of scope here: a truncated forwarded
    ``[ESCALATE]`` query on the escalation path itself is a separate,
    recorded residual).

    Ordering is load-bearing: the trim runs on the no-tag return AFTER
    :func:`~paramem.server.escalation.detect_escalation` has already run,
    so a complete ``[ESCALATE]`` tag can never be eaten by it. The
    hops-exhausted return below is NEVER trimmed regardless of
    ``is_truncated`` — its text is everything BEFORE a tag the model
    actually emitted and is therefore complete by construction.
    """
    should_escalate, forwarded_query = detect_escalation(response)

    if not should_escalate:
        text = _trim_incomplete_sentence(response) if is_truncated else response
        return ChatResult(text=text, probed_keys=probed_keys or [])

    forwarded_is_personal = is_self_referential(
        forwarded_query,
        personal_referent_config=config.personal_referent,
    )

    intent_label = intent.value if intent is not None else "unknown"
    if forwarded_is_personal:
        logger.info(
            "[ESCALATE] → HA suppressed (intent=%s): forwarded query is personal", intent_label
        )
    else:
        logger.info("[ESCALATE] → HA (intent=%s): %s", intent_label, forwarded_query[:100])
        result = _escalate_to_ha_agent(forwarded_query, ha_client, config, language=language)
        if result is not None:
            return result
    cloud_result = answer_via_cloud(
        forwarded_query,
        cloud_agent,
        config,
        is_personal=is_personal or forwarded_is_personal,
        model=model,
        tokenizer=tokenizer,
        speaker=speaker,
        speaker_id=speaker_id,
        history=history,
        language=language,
    )
    if cloud_result is not None:
        logger.info(
            "[ESCALATE] → cloud fallback (intent=%s): %s", intent_label, forwarded_query[:100]
        )
        return cloud_result

    # All escalation paths exhausted — return pre-escalation text from local model
    local_text = response.split("[ESCALATE]")[0].strip()
    return ChatResult(text=local_text or "I'm not sure about that.", probed_keys=probed_keys or [])


def _build_messages(
    text: str,
    history: list[dict] | None,
    system_prompt: str,
    tokenizer,
) -> list[dict]:
    """Build chat messages enforcing strict user/assistant alternation.

    Mistral requires: system → user → assistant → user → ...
    History read back from ``SessionBuffer`` can itself be non-alternating:
    the user and assistant turns of a single exchange are appended as two
    separate, non-atomic ``SessionBuffer.append`` calls (see
    ``paramem.server.app._run_chat_turn``) — if the assistant append fails
    after the user append succeeds, a lone user turn survives and the next
    request's history carries two consecutive user turns.  We enforce the
    pattern here regardless of how it arose.
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
