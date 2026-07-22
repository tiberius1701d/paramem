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
   config, no encoder/exemplars loaded, below-margin confidence) is
   treated as personal at the ``is_personal`` gate below (fail
   closed): Cloud is never reached.  HA stays reachable as a tool
   fallback.  Tier selection resolves ``UNKNOWN`` separately per
   :class:`IntentConfig.fail_closed_intent` (router-side); this gate
   controls only the cloud-escalation boundary.

Speaker scoping (``RoutingPlan.steps``) is the privacy boundary — only
the resolved speaker's keys can populate ``keys_to_probe``.

There is ONE personal verdict, computed once in :func:`handle_chat`: the
union of the intent classifier's ``PERSONAL``/``UNKNOWN`` result and
:func:`~paramem.server.sanitizer.check_personal_content`'s findings.  It
travels the call tree as ``is_personal`` and gates the CLOUD leg only —
HA is local and stays reachable as a tool fallback on every path.
The one exception is the model-authored forwarded query behind
``[ESCALATE]``: it is a different artifact from the turn, so
:func:`_maybe_escalate` computes a second verdict on it with the same
predicate and suppresses BOTH hops (``ha_agent_id`` is operator-pointed
and may be cloud-backed) when that verdict is personal.
:func:`answer_via_cloud` is the sole cloud-egress funnel in local mode.
:func:`_escalate_to_cloud` is a shared primitive, not a funnel — the only
other caller is ``paramem.server.app._cloud_only_route``, which runs with
no local model and therefore no ParaMem-held knowledge to protect.

Fallback chain at every escalation point: HA → cloud → local base model
(``_base_model_answer``).  ``_escalate_to_ha_agent`` is HA-only;
callers own the cloud fallback.
"""

import json
import logging
from dataclasses import dataclass, field

from paramem.cloud.providers.base import CloudAgent
from paramem.evaluation.recall import generate_answer
from paramem.graph.prompts import _load_speaker_directive_section
from paramem.models.loader import adapt_messages, grad_checkpointing_disabled
from paramem.server.config import ServerConfig
from paramem.server.escalation import detect_escalation
from paramem.server.router import Intent, RoutingPlan
from paramem.server.sanitizer import check_personal_content
from paramem.server.tools.ha_client import HAClient
from paramem.training.thermal_throttle import wait_for_cooldown as _wait_for_cooldown
from paramem.utils.identity import canonical, is_speaker_id

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10

# Loaded once at module import time; fails loud if speaker_directive.txt is missing.
THIRD_PARTY_DESCRIPTOR: str = _load_speaker_directive_section("THIRD-PARTY-DESCRIPTOR")


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
    speaker: str | None,
    language: str | None,
    config: ServerConfig | None,
) -> str:
    """Assemble the speaker + language prefix for a system prompt.

    This is the single shared implementation for the "You are speaking with X"
    + language instruction block used by both local inference and cloud
    escalation paths.

    Speaker-id-to-name resolution at inference time is handled by the
    ``speaker_resolver`` injected into :func:`MemoryStore.probe` — the raw
    ``speaker{N}`` token in recalled facts is replaced with the display name at
    the fact-render boundary, not via a prompt injection.  No id-mapping
    sentence is emitted here; none reaches the cloud (privacy invariant).

    Args:
        speaker: Display name of the resolved speaker, or ``None`` when
            unknown or suppressed (e.g. anonymous profile).
        language: BCP-47 language code, or ``None`` / ``"en"`` when no
            instruction is needed.
        config: Server config, used to derive the language display name via
            ``config.tts.language_name``.

    Returns:
        A prefix string (possibly empty) ready to be prepended to the base
        system prompt.
    """
    parts: list[str] = []
    if speaker:
        parts.append(f"You are speaking with {speaker}.")
    lang_instr = _language_instruction(language, config)
    if lang_instr:
        parts.append(lang_instr)
    return " ".join(parts)


def _personalize_prompt(
    base_prompt: str,
    speaker: str | None,
    language: str | None = None,
    config: ServerConfig | None = None,
) -> str:
    """Inject speaker name and language instruction into the system prompt.

    Uses :func:`_build_speaker_prefix` to assemble the prefix so that the
    local-inference and cloud-escalation paths share exactly one implementation.

    Speaker-id-to-name resolution is handled at the fact-render boundary via
    the ``speaker_resolver`` in :func:`MemoryStore.probe` — not by a prompt
    injection.  No ``speaker_id`` param is needed or accepted here.

    Greeting is handled at the app layer (prepended to response text)
    so it works across all paths including escalation.

    Args:
        base_prompt: The base system prompt to prepend to.
        speaker: Display name of the resolved speaker, or ``None``.
        language: BCP-47 language code, or ``None`` / ``"en"``.
        config: Server config for language display names.
    """
    prefix = _build_speaker_prefix(speaker, language, config)
    if prefix:
        return prefix + " " + base_prompt
    return base_prompt


@dataclass
class ChatResult:
    text: str
    escalated: bool = False
    probed_keys: list[str] = field(default_factory=list)


def _abstain_if_applicable(
    text: str,
    config: ServerConfig,
    *,
    is_personal: bool,
    speaker_id: str | None = None,
    router=None,
) -> tuple[ChatResult, str] | None:
    """Decide whether to short-circuit with the canned abstention response.

    Gate: ``config.abstention.enabled`` AND ``is_personal`` AND the query
    is interrogative (per :func:`paramem.server.router._is_interrogative`).
    When the gate fires, returns ``(canned_chat_result, exit_via_label)``;
    otherwise returns ``None`` and the caller continues the escalation
    chain.

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
    from paramem.server.router import _is_interrogative

    if not (
        config.abstention.enabled
        and is_personal
        and _is_interrogative(text, config=config.sentence_type)
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
    known_entities: set[str] | None = None,
    effective_mode: str | None = None,
    memory_store=None,
    speaker_store=None,
) -> ChatResult:
    """Process a chat message via intent-keyed dispatch.

    Routing reads ``RoutingPlan.intent`` populated by the router's
    classify_intent() pass:

    * ``PERSONAL`` → local PA probe + reason.  HA is reachable from the
      local model via ``[ESCALATE]`` and from the no-layers branch as a
      tool fallback.  **cloud is never reached** — personal-class queries
      stay off the cloud (privacy invariant, threaded as ``is_personal``
      through the call tree).
    * ``COMMAND`` / ``GENERAL`` → HA first (tools, live state), cloud
      fallback (reasoning).
    * ``UNKNOWN`` — intent could not be established; ``is_personal``
      treats it the same as ``PERSONAL`` (fail closed), so it never
      reaches cloud even when the routing plan carries no probe steps.

    ``is_personal`` is computed ONCE here, as the union of that intent
    verdict and :func:`~paramem.server.sanitizer.check_personal_content`'s
    findings over ``text``.  A ``COMMAND``/``GENERAL`` turn that names a
    stored entity is therefore personal too, even though the classifier
    said otherwise.  The verdict gates the CLOUD leg only; HA stays
    reachable on every path, and a personal turn that neither HA nor the
    cloud answered falls to abstention before the base model.

    The ``is_residual`` diagnostic tracks "did any graph signal fire?"
    for the routing-quality metric independent of the intent decision —
    ``True`` when neither PA steps nor HA domains were produced.

    When ``config.debug`` is True a per-request routing-decision
    diagnostic is emitted via ``logging.info(extra={"routing": …})`` at
    function exit.
    """
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
            # First arm of the personal verdict.  Fail closed: an intent that
            # could not be established (no IntentConfig, classifier
            # unavailable, below-margin confidence) counts as personal.
            intent_is_personal = intent in (Intent.PERSONAL, Intent.UNKNOWN)

            # Second arm's ground truth, assembled once per /chat call.
            # Personal-content detection is anchored on the graph's
            # subject/object names (read directly from the MemoryStore — the
            # same source the router uses for speaker scoping) plus the
            # speaker's display name (M3 coverage: the display name must be
            # flagged as a personal referent even when it is no longer a
            # registry subject under the id-as-subject refactor) plus a
            # first-person token-set + the resolved speaker_id — the same
            # ground truth the extraction-path anonymizer uses, no static
            # keyword list.  The set is rebuilt per /chat call; cost is O(N)
            # over active keys (~hundreds in production).
            #
            # Names are stripped but NOT case-folded: the sanitizer's
            # known-entity scrub matches exact-case, whole-word (case is the
            # only signal separating the person "Bill" from an electricity
            # "bill"), so it needs the display surface as stored.
            if known_entities is None and memory_store is not None:
                _entity_names: set[str] = set()
                for _tier, _key, _entry in memory_store.iter_entries():
                    for _field in ("subject", "object"):
                        _name = _entry.get(_field, "")
                        if _name and len(_name) > 1:
                            _entity_names.add(_name.strip())
                known_entities = _entity_names
            # M3: ensure the speaker's display name is always a personal-referent
            # signal regardless of what subjects appear in the registry.  Under
            # the id-as-subject convention the display name leaves the registry
            # subjects; sourcing it here from the resolved ``speaker`` argument
            # (the caller-resolved display name, None for anonymous) keeps the
            # sanitizer coverage intact without coupling inference.py to
            # SpeakerStore.
            if speaker and len(speaker) > 1:
                if known_entities is None:
                    known_entities = set()
                known_entities = known_entities | {speaker.strip()}
            # Extend known_entities with non-anonymous household display names so
            # the sanitizer covers all enrolled speakers, not just the active one.
            # household_display_names() filters out anonymous profiles
            # (enroll_method == "anonymous_voice") — only real-name disclosures are
            # added.  This closes the gap where a fact mentioning another household
            # member's display name was not recognised as personal content.
            if speaker_store is not None:
                _household_names = speaker_store.household_display_names()
                if _household_names:
                    if known_entities is None:
                        known_entities = set()
                    known_entities = known_entities | {n.strip() for n in _household_names if n}

            # Build the speaker resolver closure once per request.  Passed into
            # memory_store.probe so raw speaker{N} tokens in recalled facts are
            # replaced with display names at the render boundary.
            # read-tolerance: canonicalize before lookup so pre-migration cased
            # tokens (e.g. "Speaker0") still resolve during the forward-only
            # transition.
            def _speaker_resolver(tok: str) -> str:
                if not is_speaker_id(tok):
                    return tok
                name = speaker_store.resolve_speaker_name(canonical(tok)) if speaker_store else None
                return name if name else THIRD_PARTY_DESCRIPTOR

            # THE personal verdict, computed once.  Two arms, unioned: the
            # intent classifier (covers a first-person query naming nothing
            # the graph knows) and the graph-anchored content check (covers a
            # query naming a stored entity with no first-person marker).
            # Everything downstream reads ``is_personal``; neither arm is
            # re-derived anywhere else.
            personal_findings = check_personal_content(
                text,
                speaker_id=speaker_id,
                known_entities=known_entities,
                personal_referent_config=config.personal_referent,
            )
            is_personal = intent_is_personal or bool(personal_findings)
            routing_diags["personal_findings"] = personal_findings

            # Anonymous deny-by-default: an unauthenticated caller has no claim
            # on the speaker's private parametric memory.  When the router
            # classified the turn as PERSONAL but ``speaker_id`` did not
            # resolve, the personal probe path is unreachable for this
            # caller.  Interrogative form → canned abstention (avoids
            # leaking the existence of indexed facts via an answer).
            # Declarative form → demote to non-personal so the turn flows
            # to the General/Unknown HA → cloud path with cloud-side
            # sanitization, instead of consulting the local store.  Pairs
            # with the sanitizer paraphrase pass (Task #13) to keep
            # CV-derived topics off the personal arm for anonymous callers.
            if is_personal and not speaker_id:
                routing_diags["paths_attempted"].append("anonymous_personal_deny")
                abstention = _abstain_if_applicable(
                    text,
                    config,
                    is_personal=True,
                    speaker_id=None,
                    router=router,
                )
                if abstention is not None:
                    result, label = abstention
                    routing_diags["exit_via"] = label
                    logger.info(
                        "Anonymous caller + PERSONAL intent — abstaining (%s)",
                        label,
                    )
                    return result
                # Declarative turn from anonymous caller: do NOT consult the
                # personal store; relay via the standard escalation chain.
                is_personal = False

            # PERSONAL → local PA probe + reason.  No cloud anywhere on this
            # path: is_personal=True suppresses every internal _escalate_to_cloud
            # call (no-layers branch, post-reason [ESCALATE], base-model
            # fallthrough).  HA stays reachable as a tool fallback.
            if is_personal and plan is not None and plan.steps:
                routing_diags["paths_attempted"].append("personal")
                routing_diags["exit_via"] = "personal_probe"
                # T2c: pre-task GPU cooldown gate — wait until GPU is cool before
                # the local-PA probe + reason burst.  Placed here (after routing
                # has committed to the local PA path) so HA/cloud-routed requests
                # never wait.  Bounded by cooldown_gate_max_wait_inference_s
                # (default 30 s) — the caller proceeds with a WARNING on timeout.
                _wait_for_cooldown(
                    config.vram.cooldown_gate_threshold_c,
                    config.vram.cooldown_gate_max_wait_inference_s,
                    config.vram.cooldown_gate_poll_s,
                    label="inference",
                )
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
                    speaker_store=speaker_store,
                    speaker_resolver=_speaker_resolver,
                    known_entities=known_entities,
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
                known_entities=known_entities,
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
                known_entities=known_entities,
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


def _sanitize_history(
    history: list[dict] | None,
    known_entities: set[str] | None = None,
    *,
    speaker_id: str | None = None,
) -> list[dict]:
    """Drop-gate conversation history for cloud: personal turns are removed.

    Unconditional — there is no pass-through or warn-only setting.  A
    history turn that :func:`~paramem.server.sanitizer.check_personal_content`
    flags never egresses, whether or not the current turn is being
    placeholdered for privacy.

    Args:
        history: Conversation turns to gate.  Only the last
            :data:`MAX_HISTORY_TURNS` are considered; empty-text turns are
            dropped.
        known_entities: Optional set of **real-case** entity/speaker names to
            treat as personal referents.  When provided, household display
            names are recognised as personal content and the turn naming one
            is dropped.
        speaker_id: Resolved speaker store ID, threaded to
            :func:`~paramem.server.sanitizer.check_personal_content`'s
            first-person detector (:func:`~paramem.server.sanitizer.
            _is_about_speaker`) — without it, "I" / "my" in a history
            turn never resolves to a concrete speaker and the detector
            is dead on this channel.

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
        findings = check_personal_content(
            text, speaker_id=speaker_id, known_entities=known_entities
        )
        if findings:
            logger.info("Dropped personal-content history turn from cloud payload: %s", findings)
            continue
        sanitized.append({"role": role, "text": text})
    return sanitized


def answer_via_cloud(
    text: str,
    cloud_agent: CloudAgent | None,
    config: ServerConfig,
    *,
    is_personal: bool,
    model=None,
    tokenizer=None,
    speaker: str | None = None,
    speaker_id: str | None = None,
    history: list[dict] | None = None,
    language: str | None = None,
    known_entities: set[str] | None = None,
) -> ChatResult | None:
    """Apply the configured cloud-egress policy and call cloud accordingly.

    Returns the cloud result on success, or ``None`` when policy or per-query
    safety blocks the call (caller falls through to the next mechanism in the
    escalation chain — typically the base model or abstention).

    Policy matrix from ``config.sanitization.cloud_mode``:

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

    ``model`` and ``tokenizer`` are required when ``cloud_mode`` selects
    anonymization; they're ignored in ``block`` mode.  Passing ``None``
    in an anonymizing mode is treated as a per-query block.
    """
    if cloud_agent is None:
        return None

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
        if model is None or tokenizer is None:
            logger.warning(
                "cloud_mode=%s requires model/tokenizer for anonymization; blocking", cloud_mode
            )
            return None
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
        # doesn't do reliably).  Instead: (i) drop-gate each turn,
        # speaker_id threaded so the first-person detector actually fires
        # on this channel, then (ii) substitute through ``payload.forward``
        # — no second LLM call.  Accepted residual: ``payload.forward``
        # only covers entities the anonymizer named in the CURRENT turn,
        # so a personal entity appearing ONLY in history is dropped by the
        # gate but not placeholdered.  See benchmarking.md.
        drop_gated_history = _sanitize_history(
            history, known_entities=known_entities, speaker_id=speaker_id
        )
        sanitized_history = [
            {**turn, "text": _substitute_whole_words(turn["text"], payload.forward)}
            for turn in drop_gated_history
        ]

        result = _escalate_to_cloud(
            payload.anon_transcript,
            cloud_agent,
            config,
            speaker=speaker,
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
    sanitized_history = _sanitize_history(
        history, known_entities=known_entities, speaker_id=speaker_id
    )
    return _escalate_to_cloud(
        text,
        cloud_agent,
        config,
        speaker=speaker,
        sanitized_history=sanitized_history,
        language=language,
    )


def _escalate_to_cloud(
    text: str,
    cloud_agent: CloudAgent,
    config: ServerConfig,
    speaker: str | None = None,
    sanitized_history: list[dict] | None = None,
    language: str | None = None,
) -> ChatResult:
    """Route to cloud model for reasoning-heavy queries.

    Passes conversation history so the cloud model can derive persona,
    tone, and style from the conversation context.

    Args:
        text: The query text (already sanitized/anonymized by the caller).
        cloud_agent: Cloud agent to delegate to.
        config: Server config.
        speaker: Display name of the resolved speaker, or ``None``.
        sanitized_history: Conversation history turns, ALREADY drop-gated
            (and, under an anonymizing ``cloud_mode``, placeholdered) by the
            caller — this function does not sanitize.  The caller is the one
            place that knows the egress policy.  This is a shared primitive,
            not an egress funnel: :func:`answer_via_cloud` is the funnel in
            local mode, ``paramem.server.app._cloud_only_route`` its
            cloud-only counterpart, and nothing else calls this.
        language: BCP-47 language code.
    """
    sanitized_history = sanitized_history or []

    prefix = _build_speaker_prefix(speaker, language, config)
    prompt = (prefix + " " + CLOUD_PROMPT) if prefix else CLOUD_PROMPT

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
    speaker_store=None,
    speaker_resolver=None,
    known_entities: set[str] | None = None,
) -> ChatResult:
    """Probe adapters in memory hierarchy order, assemble layered context.

    Builds a ``keys_by_adapter`` dict from ``plan.steps`` (preserving router
    order: procedural → episodic → semantic → session adapters newest-first),
    dispatches to ``MemoryStore.probe`` for cache resolution + on-miss source
    delegation, then reassembles per-layer facts for context augmentation.

    Cache hits return in O(1).  On cache miss the mode-appropriate
    :class:`MemorySource` resolves the entry (``WeightMemorySource`` in train
    mode, ``DiskMemorySource`` in simulate mode) and the result is memoized
    back into the cache when ``config.inference.preload_cache`` is True.

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
    from peft import PeftModel

    from paramem.memory.source import (
        DiskMemorySource,
        WeightMemorySource,
    )
    from paramem.models.loader import switch_adapter

    registry = _load_simhash_registry(config.adapter_dir)

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

    # Mode-aware on-miss source.  Simulate mode persists facts to disk via
    # graph.json (DiskMemorySource).  Train mode persists facts in adapter
    # weights (WeightMemorySource — probes the weights for the entry).  The
    # MemoryStore cache is RAM-only and is the fast path; the source is
    # the slow-path fallback when a key isn't already cached.
    _active_mode = effective_mode if effective_mode else config.consolidation.mode
    if _active_mode == "simulate":
        source = DiskMemorySource(config.adapter_dir)
    elif model is not None:
        source = WeightMemorySource(
            model,
            tokenizer,
            registry=registry,
            batch_size=config.consolidation.recall_probe_batch_size,
        )
    else:
        source = None

    probe_results = memory_store.probe(
        keys_by_adapter,
        source=source,
        speaker_id=speaker_id,
        memoize=config.inference.preload_cache,
        speaker_resolver=speaker_resolver,
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
            known_entities=known_entities,
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
            known_entities=known_entities,
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
        (n for n in layers if n == "episodic" or n.startswith("episodic_interim_")),
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
        known_entities=known_entities,
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
    known_entities: set[str] | None = None,
) -> ChatResult:
    """Answer from base model without context — escalation candidate.

    ``is_personal`` propagates the privacy gate to ``_maybe_escalate`` so
    a base-model [ESCALATE] from a personal-class query cannot reach
    Cloud.  ``known_entities`` is forwarded for the same reason: it is one
    of the two arms ``_maybe_escalate`` uses to give the model-authored
    forwarded query its own personal verdict.
    """
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
        cloud_agent=cloud_agent,
        ha_client=ha_client,
        speaker=speaker,
        speaker_id=speaker_id,
        history=history,
        language=language,
        is_personal=is_personal,
        model=model,
        tokenizer=tokenizer,
        known_entities=known_entities,
    )


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
    known_entities: set[str] | None = None,
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
    here with the same :func:`~paramem.server.sanitizer.check_personal_content`
    predicate that produced the turn verdict.  A personal forwarded query
    suppresses the HA hop outright (``ha_agent_id`` is operator-configurable
    and is routinely pointed at a cloud-backed agent), and is unioned into
    the ``is_personal`` passed to :func:`answer_via_cloud` so the existing
    ``cloud_mode`` policy applies to the stronger of the two verdicts.

    ``model`` and ``tokenizer`` are forwarded to
    :func:`answer_via_cloud` so the anonymizer (when
    selected) can rewrite outbound text.
    """
    should_escalate, forwarded_query = detect_escalation(response)

    if not should_escalate:
        return ChatResult(text=response, probed_keys=probed_keys or [])

    forwarded_is_personal = bool(
        check_personal_content(
            forwarded_query,
            speaker_id=speaker_id,
            known_entities=known_entities,
            personal_referent_config=config.personal_referent,
        )
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
        known_entities=known_entities,
    )
    if cloud_result is not None:
        logger.info(
            "[ESCALATE] → cloud fallback (intent=%s): %s", intent_label, forwarded_query[:100]
        )
        return cloud_result

    # All escalation paths exhausted — return pre-escalation text from local model
    local_text = response.split("[ESCALATE]")[0].strip()
    return ChatResult(text=local_text or "I'm not sure about that.", probed_keys=probed_keys or [])


def _load_simhash_registry(adapter_dir) -> dict:
    """Load combined SimHash dict by merging per-adapter indexed_key_registry.json files.

    Returns ``{key: simhash}`` across all main and interim adapter slots.

    Reads the ``"simhash"`` map from each tier's
    ``<adapter_dir>/<tier>/indexed_key_registry.json`` for the three main
    tiers (episodic, semantic, procedural) and from each interim slot under
    ``<adapter_dir>/episodic/interim_<stamp>/indexed_key_registry.json``.

    The registry file is the single source of truth for per-key fingerprints
    (active∪stale superset) since the SimHash unification refactor.  The
    separate ``simhash_registry.json`` sidecar is no longer written.

    When a key appears in multiple tier files (a transient state during
    promotion) the later read wins — the content is the same regardless of
    which tier holds the key.
    """
    from pathlib import Path as _Path

    registry: dict = {}
    adapter_dir = _Path(adapter_dir)
    if not adapter_dir.exists():
        return registry

    from paramem.backup.encryption import read_maybe_encrypted

    def _merge_registry_file(p: _Path) -> None:
        """Extract the ``"simhash"`` map from one indexed_key_registry.json."""
        try:
            raw = json.loads(read_maybe_encrypted(p).decode("utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read registry file %s — skipping", p.name)
            return
        if not isinstance(raw, dict):
            return
        simhash_map = raw.get("simhash", {})
        if not isinstance(simhash_map, dict):
            return
        for key, fp in simhash_map.items():
            if isinstance(fp, int):
                registry[key] = fp

    # Per-tier main paths.
    for tier in ("episodic", "semantic", "procedural"):
        p = adapter_dir / tier / "indexed_key_registry.json"
        if p.exists():
            _merge_registry_file(p)

    # Interim adapter slots.
    from paramem.memory.interim_adapter import iter_interim_dirs

    for _name, interim_dir in iter_interim_dirs(adapter_dir):
        p = interim_dir / "indexed_key_registry.json"
        if p.exists():
            _merge_registry_file(p)

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
