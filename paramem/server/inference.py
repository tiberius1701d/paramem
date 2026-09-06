"""Chat inference — intent-driven local routing dispatch.

Dispatch is on ``RoutingPlan.intent`` populated by the router:

1. ``PERSONAL`` → local adapter probe + base-model reasoning
   (``_probe_and_reason``); if the local model emits ``[ESCALATE]``,
   the forwarded query flows through the HA door then the cloud door per
   :func:`_maybe_escalate`.
2. ``COMMAND`` → HA door first (verbatim sanitized query), cloud door
   fallback only when HA is unreachable.
3. ``GENERAL`` → HA door first, cloud door fallback.
4. ``UNKNOWN`` (intent could not be established — no classifier
   config, no encoder/exemplars loaded, below-margin confidence, or an
   unparseable LLM verdict) is **not** positively PERSONAL: it grants
   no personal-memory access and does not block cloud escalation —
   routed identically to ``GENERAL`` (HA door first, cloud door fallback).
   Only a positive PERSONAL verdict reaches the local parametric-memory
   probe branch.

Speaker scoping (``RoutingPlan.steps``) is the privacy boundary — only
the resolved speaker's keys can populate ``keys_to_probe``.

There is ONE personal verdict, computed once in :func:`handle_chat`: the
union of the intent classifier's ``PERSONAL`` verdict and
:func:`~paramem.server.sanitizer.is_self_referential`'s verdict on the
raw text.  It travels the call tree as ``is_personal`` and gates the
CLOUD leg and the choice of the local parametric-memory probe branch — the
HA leg stays reachable on every path and carries the turn verbatim
regardless of the verdict.  The one exception is
the model-authored forwarded query behind ``[ESCALATE]``: it is a
different artifact from the turn, so :func:`_maybe_escalate` computes a
second verdict on it with the same predicate and suppresses the HA hop
(``ha_agent_id`` is operator-pointed and may be cloud-backed) when that
verdict is personal.

This module owns local routing only — the personal probe, the base-model
reasoning generate, the escalation-tag detection, and the plumbing around
them (``_build_system_prompt`` / ``_build_speaker_prefix`` /
``_build_messages``, the abstention gate, the trim helpers, the context
renderers). What leaves the house is decided by
:mod:`paramem.server.egress`: :func:`~paramem.server.egress.answer_via_ha`
and :func:`~paramem.server.egress.answer_via_cloud` are the two doors this
module calls at every escalation point, each taking an
:class:`~paramem.server.egress.OutboundText` built once per outbound text.
Both doors return ``None`` when their leg did not answer, so every caller
here keeps the same fall-through chain: HA door → cloud door → abstention →
local base model (``_base_model_answer``).
"""

import logging
from datetime import date
from typing import Any

from paramem.cloud.providers.base import CloudAgent
from paramem.evaluation.recall import generate_answer
from paramem.graph.phase_trace import extraction_trace, phase_trace
from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX
from paramem.models.loader import (
    base_model_inference,
    grad_checkpointing_disabled,
    render_chat_prompt,
)
from paramem.server.chat_result import ChatResult
from paramem.server.config import ServerConfig
from paramem.server.egress import LEG_NAMES, OutboundText, answer_via_cloud, answer_via_ha
from paramem.server.escalation import detect_escalation
from paramem.server.prompts import (
    empty_period_note,
    identity_line,
    language_instruction,
    reasoning_turn,
    recorded_dates_suffix,
    serving_system_prompt,
)
from paramem.server.router import Intent, RoutingPlan, RoutingStep
from paramem.server.sanitizer import is_self_referential
from paramem.server.session_buffer import MAX_HISTORY_TURNS
from paramem.server.temporal import build_date_by_key, weekday_name
from paramem.server.temporal_selection import select_date_groups
from paramem.server.tools.ha_client import HAClient
from paramem.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# The one phase name the serving dispatch (handle_chat) opens — see
# paramem.graph.phase_trace.PHASE_NAMES and its doc table.
_SERVING_PHASE = "serve_turn"


def _leg_open(forced_leg: str | None, leg: str) -> bool:
    """Whether *leg* may be tried this turn.

    ``forced_leg`` is the probe door's route selection
    (``None`` on every production ``/chat``/``/voice`` turn — every leg is
    open). A non-``None`` value selects exactly one leg; forcing never
    bypasses that leg's own policy (``cloud_mode``, the personal verdict,
    ``cloud_permitted`` all still apply).

    Args:
        forced_leg: ``None``, or one of :data:`~paramem.server.egress.LEG_NAMES`.
        leg: The leg being considered — one of
            :data:`~paramem.server.egress.LEG_NAMES`.

    Returns:
        ``True`` iff *forced_leg* is ``None`` or equals *leg*.

    Raises:
        ValueError: If *forced_leg* is not ``None`` and not a member of
            :data:`~paramem.server.egress.LEG_NAMES`.
    """
    if forced_leg is not None and forced_leg not in LEG_NAMES:
        raise ValueError(
            f"_leg_open: unknown forced_leg {forced_leg!r}; expected one of {LEG_NAMES}"
        )
    return forced_leg is None or forced_leg == leg


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
    :func:`_probe_and_reason` and :func:`_base_model_answer`).  The cloud
    transport primitive (:func:`~paramem.server.egress._escalate_to_cloud`)
    never calls it and carries no identity line at all — cloud never
    learns the speaker's id or name.

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
        parts.append(identity_line(speaker_id))
    lang_instr = language_instruction(language, config)
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
    :func:`~paramem.server.prompts.serving_system_prompt`.  Both
    :func:`_probe_and_reason` and :func:`_base_model_answer` call this —
    collapsing what were previously two byte-identical four-line blocks —
    so the two legs cannot drift from each other.

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
        config: Server config — supplies the language display name via
            ``config.tts.language_name``.

    Returns:
        The complete system prompt: the identity/language prefix (when any)
        followed by the base serving prompt.
    """
    prefix = _build_speaker_prefix(speaker_id, language, config)
    base_prompt = serving_system_prompt()
    return f"{prefix} {base_prompt}" if prefix else base_prompt


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
    memory_store,
    router=None,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    effective_mode: str | None = None,
    forced_leg: str | None = None,
) -> ChatResult:
    """Process a chat message via intent-keyed dispatch.

    ``memory_store`` is required: in production it is ``_state["memory_store"]``,
    assigned unconditionally at lifespan (both local and cloud-only boot).

    Routing reads ``RoutingPlan.intent`` populated by the router's
    classify_intent() pass:

    * ``PERSONAL`` → local PA probe + reason.  The HA door is reachable
      from the local model via ``[ESCALATE]`` and from the no-layers
      branch.  **cloud is never reached** — personal-class queries stay
      off the cloud (privacy invariant, threaded as ``is_personal``
      through the call tree).
    * ``COMMAND`` / ``GENERAL`` / ``UNKNOWN`` → HA door first (tools, live
      state), cloud door fallback (reasoning).  ``UNKNOWN`` (intent could
      not be established) is not positively PERSONAL, so it is treated
      identically to ``GENERAL`` — no personal-memory access, and cloud
      escalation stays available.

    ``is_personal`` is computed ONCE here, as the union of that intent
    verdict and :func:`~paramem.server.sanitizer.is_self_referential`'s
    verdict over ``text``.  A ``COMMAND``/``GENERAL`` turn that refers to
    the speaker is therefore personal too, even though the classifier
    said otherwise.  The verdict gates the CLOUD leg and selects the
    local parametric-memory probe branch; the HA leg stays reachable on
    every path and carries the turn verbatim regardless of
    the verdict, and a personal turn that neither HA nor the cloud
    answered falls to abstention before the base model.

    The ``is_residual`` diagnostic tracks "did any graph signal fire?"
    for the routing-quality metric independent of the intent decision —
    ``True`` when neither PA steps nor HA domains were produced.  It is
    computed unconditionally (not gated on ``config.debug``) since it is
    part of the returned ``ChatResult.diagnostics``, not just a log field.

    The whole dispatch runs inside its own ``paramem.graph.phase_trace``
    scope (``extraction_trace()`` + ``phase_trace("serve_turn")``), so a
    production turn always has a trace to record onto even with no
    calibration caller — this is also what makes the returned
    diagnostics dict the same object recorded as the phase's ``parsed``
    field. When ``config.debug`` is True the routing decision is
    additionally emitted via ``logging.info(extra={"routing": …})`` at
    function exit.

    Args:
        forced_leg: The probe door's resolved route (``None`` on every
            production ``/chat``/``/voice`` turn — the routed dispatch
            below), or ``"ha"``/``"cloud"`` to select exactly one leg.
            Forcing a leg never bypasses that leg's own policy.

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
        "exit_via": None,
        "is_residual": False,
    }

    def _dispatch() -> ChatResult:
        with grad_checkpointing_disabled(model):
            plan = None

            # Dual-graph entity routing.
            if router is not None:
                plan = router.route(text, speaker_id=speaker_id)
            if plan is not None:
                routing_diags["intent"] = plan.intent.value

            intent = plan.intent if plan is not None else Intent.UNKNOWN

            # is_residual: the routing plan landed with no probe steps and
            # no HA domains — i.e. no deterministic signal fired and the
            # classifier residual (encoder cosine or LLM generate) drove
            # the intent verdict.  Tracks whether the routing-quality
            # metric should count this query toward the residual
            # classifier's evaluation.  Computed here, unconditionally —
            # not gated on config.debug — because it is part of the
            # returned diagnostics artifact now, not just a log field.
            routing_diags["is_residual"] = bool(
                plan is not None and not plan.steps and not plan.ha_domains
            )

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
            # path: is_personal=True suppresses every internal cloud-door
            # call (no-layers branch, post-reason [ESCALATE], base-model
            # fallthrough).  The HA door stays reachable.  Only entered when
            # no leg is forced — a forced probe route selects a leg, not the
            # personal probe branch.
            if forced_leg is None and is_personal and plan is not None and plan.steps:
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
            # steps path, and a forced-probe turn) → HA door first, cloud
            # door fallback.  is_personal still gates cloud so a defensive
            # PERSONAL request never reaches the cloud.
            intent_label = intent.value
            routing_diags["paths_attempted"].append(intent_label)
            outbound = OutboundText(
                text,
                config,
                diagnostics=routing_diags,
                history=history,
                model=model,
                tokenizer=tokenizer,
                speaker=speaker,
                speaker_id=speaker_id,
                language=language,
                is_personal=is_personal,
            )
            if _leg_open(forced_leg, "ha"):
                logger.info("Intent dispatch: %s → HA door first", intent_label)
                result = answer_via_ha(outbound, ha_client)
                if result is not None:
                    routing_diags["exit_via"] = f"{intent_label}_ha"
                    return result
            if _leg_open(forced_leg, "cloud"):
                cloud_result = answer_via_cloud(outbound, cloud_agent)
                if cloud_result is not None:
                    routing_diags["exit_via"] = f"{intent_label}_cloud"
                    logger.info("HA door produced nothing, routing to cloud door")
                    return cloud_result

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
                diagnostics=routing_diags,
                cloud_agent=cloud_agent,
                ha_client=ha_client,
                speaker=speaker,
                speaker_id=speaker_id,
                language=language,
                is_personal=is_personal,
            )

    try:
        with extraction_trace(), phase_trace(_SERVING_PHASE) as phase:
            result = _dispatch()
            result.diagnostics.update(routing_diags)
            phase.set_raw(result.text)
            phase.set_parsed(result.diagnostics)
            return result
    finally:
        if getattr(config, "debug", False):
            logger.info("routing decision", extra={"routing": routing_diags})


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
    messages = _build_messages(text, history, system_prompt)
    prompt = render_chat_prompt(messages, tokenizer, add_generation_prompt=True)
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


def _render_tier_facts(
    facts: list[tuple[str, str]],
    *,
    date_by_key: dict[str, date | None] | None = None,
) -> str:
    """Render one tier's recalled ``(key, fact_text)`` pairs as bullet lines.

    Byte-identical to a plain ``"- {fact_text}"`` bullet list joined by
    newlines when *date_by_key* is ``None`` — the
    ``temporal_selection_enabled=False`` no-op contract; "dated" is not a
    separate flag, it is *date_by_key* being present. When a map is
    supplied, facts are grouped under ``"On {Weekday}, YYYY-MM-DD:"``
    headers (dates descending; weekday via
    :func:`paramem.server.temporal.weekday_name`, matching the shape of
    the ``"Today is {Weekday}, {YYYY-MM-DD}."`` header line built in
    :func:`_render_augmented_text`); facts whose key
    has no parseable date render first, ungrouped. *date_by_key* is the
    request's single :func:`~paramem.server.temporal.build_date_by_key`
    parse — no second bookkeeping read, no second clock read, no re-parse
    of the raw value.

    Args:
        facts: ``(key, fact_text)`` pairs recalled for one tier, in probe
            order.
        date_by_key: ``{key: parsed date or None}`` for every key probed
            this request, or ``None`` when the date-group selection stage
            did not run — reproduces the original flat rendering exactly.

    Returns:
        The rendered tier body (bullet lines, optionally under date
        headers) — without the ``"[Label]"`` section wrapper.
    """
    if date_by_key is None:
        return "\n".join(f"- {fact_text}" for _key, fact_text in facts)

    undated_lines: list[str] = []
    by_date: dict[date, list[str]] = {}
    for key, fact_text in facts:
        day = date_by_key.get(key)
        if day is None:
            undated_lines.append(f"- {fact_text}")
        else:
            by_date.setdefault(day, []).append(f"- {fact_text}")

    sections: list[str] = []
    if undated_lines:
        sections.append("\n".join(undated_lines))
    for day in sorted(by_date, reverse=True):
        sections.append(f"On {weekday_name(day)}, {day.isoformat()}:\n" + "\n".join(by_date[day]))

    return "\n".join(sections)


def _render_augmented_text(
    layered_context: str,
    text: str,
    *,
    today: date | None = None,
    note: str | None = None,
) -> str:
    """Render the reasoning-turn prompt from the assembled tier context.

    Byte-identical to the undated rendering of
    :func:`~paramem.server.prompts.reasoning_turn` (``today_prefix=""``)
    when *today* is ``None`` — the ``temporal_selection_enabled=False``
    no-op contract; "dated" is not a separate flag, it is *today* being
    present (the date-group selection stage is the only caller that
    resolves it). When *today* is given, the
    header carries its weekday and ISO date, and *note* (the deterministic
    nothing-in-period note the date-group selection stage may have built)
    renders as its own line above *layered_context* — or alone when
    *layered_context* is empty (the zero-survivor case). Rendering itself
    is a single call to :func:`~paramem.server.prompts.reasoning_turn`
    (``configs/prompts/serving_directives.txt`` § REASONING-TURN) — the
    ``today is None``/``today is given`` split only computes the
    ``today_prefix``/``context`` values fed into it, so there is exactly
    one prompt-rendering call site for both shapes.

    Args:
        layered_context: The assembled, already-rendered tier sections
            (may be empty).
        text: The user's current turn.
        today: The calendar date to render in the header, or ``None``
            when the date-group selection stage did not run.
        note: The nothing-in-period note, or ``None`` when none was
            built.

    Returns:
        The full augmented prompt text passed to the reasoning generate.
    """
    if today is None:
        today_prefix, body = "", layered_context
    else:
        today_prefix = f"Today is {weekday_name(today)}, {today.isoformat()}. "
        body = "\n\n".join(part for part in (note, layered_context) if part)
    return reasoning_turn(today_prefix=today_prefix, context=body, question=text)


def _probe_and_reason(
    text: str,
    plan: RoutingPlan,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    memory_store,
    cloud_agent: CloudAgent | None = None,
    ha_client: HAClient | None = None,
    speaker: str | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    is_personal: bool = False,
    effective_mode: str | None = None,
) -> ChatResult:
    """Probe adapters in memory hierarchy order, assemble layered context.

    ``memory_store`` is required: in production it is ``_state["memory_store"]``,
    threaded through :func:`handle_chat`.

    Builds a ``keys_by_adapter`` dict from the routing plan's steps
    (preserving router order: procedural → episodic → semantic → session
    adapters newest-first), then reads it through exactly one of the two
    serving doors, forked once on ``config.inference.preload_cache`` — the
    ``True`` arm calls :meth:`~paramem.memory.store.MemoryStore.probe_cache`
    (a plain RAM-mirror lookup, no source built); the ``False`` arm builds
    the :class:`~paramem.memory.source.MemorySource` via
    :func:`~paramem.memory.source.build_memory_source` and calls
    :meth:`~paramem.memory.store.MemoryStore.probe_source` (one grouped
    weight/disk probe, gated by the store's own SimHash confidence check).
    Results then reassemble into per-layer facts for context augmentation.

    After weight probing (the ``preload_cache=False`` arm only — the cache
    arm never touches the model), restores the model to the ``episodic``
    adapter so the next query starts from a predictable state — this
    restore runs only on the paths that actually probed the weights; the
    zero-survivor date-selection path below returns before probing and
    never touches adapter state, and the reasoning generate that follows
    (here or on that path) restores its own adapter state via
    ``base_model_inference`` regardless. The reasoning phase uses
    ``model.disable_adapter()`` so the active adapter during generation
    does not matter — only the post-probe state (restored here, when
    reached) does.

    Privacy gate: ``is_personal`` flows through to every internal cloud
    fallback site (no-layers branch, base-model fallthrough, post-reason
    [ESCALATE]).  Personal-class queries never reach the cloud.

    Date-group selection stage: when ``config.inference.temporal_selection_enabled``
    is True and ``plan.steps`` is non-empty, asks the local model
    (adapter off, temperature 0, via
    :func:`~paramem.server.temporal_selection.select_date_groups`) which of
    the plan's keys — read from bookkeeping and parsed to a calendar date
    once via :func:`~paramem.server.temporal.build_date_by_key` — the
    current turn needs, and probes only the surviving keys. This is the
    ENTIRE stage: the clock read, the bookkeeping reads, the selection
    call, the key filtering, and the dated context rendering are all
    gated on the same condition, so a disabled (or inapplicable) stage
    leaves this function's behavior — including the exact context string
    — byte-identical to the stage never having existed. When every key is
    filtered out, the probe call and the ``not layers`` escalation branch
    below are both structurally unreachable — the reasoning turn runs
    directly off a deterministic nothing-in-period note instead.

    Each result dict carries a ``fact_text`` field; per-tier rendering
    (the bullet-list "- " prefix, plus date headers when the selection
    stage ran) happens once, at context assembly
    (:func:`_render_tier_facts`), reading the same ``date_by_key`` map
    the selection stage built — no fact's date is parsed more than once.

    The returned ``ChatResult.diagnostics`` carries this leg's own keys
    (``temporal``, and — once probing actually happens — ``probes`` and,
    on the full probe-assembly path, ``facts_recalled``); see the
    key-presence contract in :class:`ChatResult`'s docstring for exactly
    which branches set which keys.
    """
    from paramem.memory.source import build_memory_source
    from paramem.models.loader import switch_adapter

    LAYER_LABELS = {
        "procedural": "Behavioral preferences",
        "semantic": "Consolidated knowledge",
        "episodic": "Recent knowledge",
    }

    # Diagnostics for this leg, stamped onto the returned ChatResult once —
    # after _run() below returns, regardless of which internal branch
    # produced the result (mirrors handle_chat's single-stamp pattern).
    # "temporal" is always present once this function runs: None when the
    # date-group selection stage did not run, a dict when it did.
    # "probes"/"facts_recalled" are conditional — see ChatResult's
    # docstring for the exact per-branch presence contract.
    diags: dict[str, Any] = {"temporal": None}

    def _run() -> ChatResult:
        # Date-group selection stage.  ``active_steps`` defaults to
        # ``plan.steps`` itself (same list, same objects) so every downstream
        # read is unaffected when the stage does not run — the no-op
        # contract for temporal_selection_enabled=False.  ``date_by_key``
        # stays None on that no-op path too, and every renderer downstream
        # treats None as "the stage didn't run" rather than taking a second
        # explicit flag.
        active_steps = plan.steps
        today: date | None = None
        date_by_key: dict[str, date | None] | None = None
        period_note: str | None = None
        temporal_stage_active = config.inference.temporal_selection_enabled and bool(plan.steps)

        if temporal_stage_active:
            today = date.today()
            last_seen_by_key: dict[str, object] = {}
            for step in plan.steps:
                for key in step.keys_to_probe:
                    bookkeeping = memory_store.bookkeeping_for_key(key)
                    last_seen_by_key[key] = bookkeeping.get("last_seen")

            # Single parse of every key's raw bookkeeping value for this
            # request — the selection inventory, the survivor filter, the
            # nothing-in-period note, and the per-fact rendering below all
            # read from this one map instead of re-parsing.
            date_by_key = build_date_by_key(last_seen_by_key)
            selection = select_date_groups(
                text,
                date_by_key,
                model=model,
                tokenizer=tokenizer,
                config=config,
                today=today,
            )

            active_steps = []
            total_before = 0
            total_after = 0
            for step in plan.steps:
                total_before += len(step.keys_to_probe)
                kept = [key for key in step.keys_to_probe if selection.selects(date_by_key[key])]
                total_after += len(kept)
                if kept:
                    active_steps.append(
                        RoutingStep(adapter_name=step.adapter_name, keys_to_probe=kept)
                    )

            # Deterministic nothing-in-period note: the selection chose
            # specific ranges (not `all`) and zero surviving keys carry a
            # parseable date — regardless of whether undated keys survived
            # via ``include_undated`` (they still probe and render alongside
            # the note).
            if not selection.all:
                dated_kept = any(
                    date_by_key[key] is not None
                    for step in active_steps
                    for key in step.keys_to_probe
                )
                if not dated_kept:
                    existing_dates = sorted(
                        {day for day in date_by_key.values() if day is not None}
                    )
                    available_dates = ", ".join(day.isoformat() for day in existing_dates)
                    period_note = empty_period_note()
                    if available_dates:
                        period_note += recorded_dates_suffix(available_dates)

            diags["temporal"] = {
                "all": selection.all,
                "ranges": len(selection.ranges),
                "include_undated": selection.include_undated,
                "fail_open": selection.fail_open,
                "keys_before": total_before,
                "keys_after": total_after,
                "period_note": period_note is not None,
            }

            logger.info(
                "Date-group selection: %s, kept %d/%d key(s)",
                (
                    "all"
                    if selection.all
                    else f"{len(selection.ranges)} range(s)"
                    + (" +undated" if selection.include_undated else "")
                )
                + (" (fail-open)" if selection.fail_open else ""),
                diags["temporal"]["keys_after"],
                diags["temporal"]["keys_before"],
            )

            if total_after == 0:
                # No keys survived selection at all: the probe call and the
                # `not layers` escalation branch below are never reached on
                # this path — the reasoning turn runs directly off the note.
                augmented_text = _render_augmented_text("", text, today=today, note=period_note)
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
                    diagnostics=diags,
                    intent=plan.intent,
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

        # Build ordered keys_by_adapter dict from routing steps.
        # Insertion order matches router output (procedural → episodic → semantic
        # → session adapters newest-first).  Use a plain dict — Python 3.7+
        # guarantees insertion-order preservation.
        keys_by_adapter: dict[str, list[str]] = {}
        for step in active_steps:
            keys_by_adapter[step.adapter_name] = list(step.keys_to_probe)

        # Fork once on inference.preload_cache — the two serving read doors
        # are exclusive, never layered.  The cache arm never builds a
        # source and never touches the model; the source arm builds it
        # here and owns the post-probe adapter restore (the cache arm's
        # reasoning generate restores its own adapter state via
        # base_model_inference regardless — nothing to undo here).
        if config.inference.preload_cache:
            probe_results = memory_store.probe_cache(keys_by_adapter)
        else:
            _active_mode = effective_mode if effective_mode else config.consolidation.mode
            source = build_memory_source(
                mode=_active_mode,
                adapter_dir=config.adapter_dir,
                batch_size=config.consolidation.recall_probe_batch_size,
                model=model,
                tokenizer=tokenizer,
                # Per-turn probe only: reuse the process-wide simhash-registry cache
                # instead of re-reading and re-parsing every tier's
                # indexed_key_registry.json from disk on every personal turn.
                # QueryRouter.reload() invalidates the cache after every
                # registry-mutating cycle, so this never serves stale fingerprints.
                cached_registry=True,
            )

            probe_results = memory_store.probe_source(keys_by_adapter, source=source)

            # Restore predictable adapter state after weight probing: episodic is
            # the main adapter for PM inference.  The reasoning phase uses
            # disable_adapter() so the active adapter during generation does not
            # matter — only the post-return state (restored here) does.  No-op in
            # simulate mode where probing didn't touch the model.
            if _active_mode != "simulate" and model is not None and "episodic" in model.peft_config:
                switch_adapter(model, "episodic")

        # Reassemble per-step facts so each adapter's results go to its layer.
        # Each fact retains its originating key (rather than a pre-rendered
        # "- {fact}" string) so the date-grouped rendering at context assembly
        # (_render_tier_facts) can look each fact's date up in date_by_key
        # without a second bookkeeping read or a re-parse.
        layers: dict[str, list[tuple[str, str]]] = {}
        diags["probes"] = {}

        for step in active_steps:
            layer_facts: list[tuple[str, str]] = []
            for key in step.keys_to_probe:
                result = probe_results.get(key)
                if result and "failure_reason" not in result:
                    # fact_text is guaranteed on every success result from
                    # probe_keys_grouped_by_adapter (train mode) / DiskMemorySource
                    # (simulate mode). The get() fallback covers mocked/legacy
                    # callers that return a bare {answer: ...} dict without the
                    # field.
                    layer_facts.append((key, result.get("fact_text", result.get("answer", ""))))

            if layer_facts:
                layers[step.adapter_name] = layer_facts

            diags["probes"][step.adapter_name] = {
                "probed": len(step.keys_to_probe),
                "recalled": len(layer_facts),
            }
            logger.info(
                "Adapter %s: probed %d keys, recalled %d facts",
                step.adapter_name,
                diags["probes"][step.adapter_name]["probed"],
                diags["probes"][step.adapter_name]["recalled"],
            )

        if not layers:
            logger.info(
                "All %d probed key(s) failed, escalating via HA%s (intent=%s)",
                sum(len(s.keys_to_probe) for s in active_steps),
                "" if is_personal else " → cloud",
                plan.intent.value,
            )
            outbound = OutboundText(
                text,
                config,
                diagnostics=diags,
                history=history,
                model=model,
                tokenizer=tokenizer,
                speaker=speaker,
                speaker_id=speaker_id,
                language=language,
                is_personal=is_personal,
            )
            result = answer_via_ha(outbound, ha_client)
            if result is not None:
                return result
            cloud_result = answer_via_cloud(outbound, cloud_agent)
            if cloud_result is not None:
                return cloud_result
            # Abstention: ``_probe_and_reason`` is reached only for PERSONAL with
            # non-empty plan.steps (handle_chat dispatch).  Probes failed and HA
            # had no tool answer either; the base model has no context here
            # (``not layers`` means no facts were recalled), so generating an
            # answer would be unconditional confabulation.
            #
            # ``router`` and ``speaker_id`` are deliberately NOT passed to this
            # ``_abstain_if_applicable`` call (both default to ``None`` here) —
            # cold-start can't apply: reaching here means the router already
            # built probes from the speaker's existing keys, so the speaker has
            # facts and the coverage-gap (canned) response fits.  ``speaker_id``
            # itself IS a parameter of ``_probe_and_reason`` and is threaded to
            # the escalation call above and to ``_base_model_answer`` below.
            abstention = _abstain_if_applicable(text, config, is_personal=is_personal)
            if abstention is not None:
                result, _label = abstention
                logger.info(
                    "Abstention: PA-empty personal interrogative in _probe_and_reason "
                    "(probes=%d failed, HA returned None)",
                    sum(len(s.keys_to_probe) for s in active_steps),
                )
                return result

            return _base_model_answer(
                text,
                history,
                model,
                tokenizer,
                config,
                diagnostics=diags,
                cloud_agent=cloud_agent,
                ha_client=ha_client,
                speaker=speaker,
                speaker_id=speaker_id,
                language=language,
                is_personal=is_personal,
            )

        total_facts = sum(len(f) for f in layers.values())
        diags["facts_recalled"] = total_facts
        logger.info("Total recalled: %d facts from %d layers", diags["facts_recalled"], len(layers))

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
            context_sections.append(
                f"[{LAYER_LABELS['procedural']}]\n"
                + _render_tier_facts(procedural_facts, date_by_key=date_by_key)
            )

        episodic_adapter_names = sorted(
            (n for n in layers if n == "episodic" or n.startswith(INTERIM_NAME_PREFIX)),
            key=lambda n: (n != "episodic", n),
            reverse=True,
        )
        episodic_facts: list[tuple[str, str]] = []
        for adapter_name in episodic_adapter_names:
            episodic_facts.extend(layers[adapter_name])
        if episodic_facts:
            context_sections.append(
                f"[{LAYER_LABELS['episodic']}]\n"
                + _render_tier_facts(episodic_facts, date_by_key=date_by_key)
            )

        semantic_facts = layers.get("semantic")
        if semantic_facts:
            context_sections.append(
                f"[{LAYER_LABELS['semantic']}]\n"
                + _render_tier_facts(semantic_facts, date_by_key=date_by_key)
            )

        layered_context = "\n\n".join(context_sections)
        augmented_text = _render_augmented_text(
            layered_context, text, today=today, note=period_note
        )

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
            diagnostics=diags,
            intent=plan.intent,
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

    result = _run()
    result.diagnostics.update(diags)
    return result


def _base_model_answer(
    text: str,
    history: list[dict] | None,
    model,
    tokenizer,
    config: ServerConfig,
    *,
    diagnostics: dict[str, Any],
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
    Cloud.  ``diagnostics`` is the caller's turn-scoped dict, threaded
    unchanged to :func:`_maybe_escalate` (and, through it, to
    :func:`~paramem.server.egress.answer_via_cloud`) so a [ESCALATE] cloud
    hop from this leg authors the egress record into the same dict the
    caller merges onto its returned :class:`ChatResult`.
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
        diagnostics=diagnostics,
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


def _pre_escalation_result(response: str) -> ChatResult:
    """Build the terminal :class:`ChatResult` from the model's pre-tag text.

    Shared by :func:`_maybe_escalate`'s two escalation-abandoned terminals
    — no forwarded query to escalate, and every escalation hop (HA, cloud)
    exhausted — both of which fall back to the SAME text: everything
    before the ``[ESCALATE]`` tag the model emitted, or a canned reply
    when that prefix is empty or whitespace-only (the model emitted the
    tag with nothing usable ahead of it).

    Args:
        response: The local model's raw generated text, tag included.

    Returns:
        A :class:`ChatResult` carrying the pre-tag text, or the canned
        "I'm not sure about that." reply when the prefix is blank.
    """
    local_text = response.split("[ESCALATE]")[0].strip()
    return ChatResult(text=local_text or "I'm not sure about that.")


def _maybe_escalate(
    response: str,
    config: ServerConfig,
    *,
    diagnostics: dict[str, Any],
    intent: Intent | None = None,
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
    """Check for [ESCALATE] tag and route through the HA door then the cloud door.

    This is the escalation-from-a-failed-local-answer path only.  Device
    control does not pass through here: an imperative with an HA entity
    match routes to HA directly in ``handle_chat`` and never reaches this
    function.

    HA agent has tools (search, device control, real-time data) so it
    gets first shot. Cloud handles queries that need pure reasoning.
    When both hops are suppressed or fail, the pre-escalation portion of
    the local response is returned (text before the [ESCALATE] marker).

    An empty ``forwarded_query`` (the model emitted the tag with nothing,
    or only whitespace/a bare ``":"``, after it) is nothing to escalate —
    neither door is consulted, and this function falls straight through to
    the pre-escalation text, the same return the hops-exhausted path below
    uses.  This is a caller-side precondition, not a cause either door can
    name: an empty transcript reaching the anonymize chain is an invariant
    violation, not a privacy decision.

    Privacy invariant: the forwarded query is a **model-authored** artifact,
    not the user's turn — on the personal path the model has already
    recalled facts from parametric memory and may have written them into
    the text after the tag.  It therefore carries its OWN verdict, computed
    here with the same :func:`~paramem.server.sanitizer.is_self_referential`
    predicate that produced the turn verdict.  A personal forwarded query
    suppresses the HA door outright (``ha_agent_id`` is operator-configurable
    and is routinely pointed at a cloud-backed agent), and is unioned into
    the ``is_personal`` passed to the cloud door so the existing
    ``cloud_mode`` policy applies to the stronger of the two verdicts.

    ``model`` and ``tokenizer`` are carried by the
    :class:`~paramem.server.egress.OutboundText` this function builds for
    the forwarded query, so the anonymizer (when selected) can rewrite
    outbound text.  ``diagnostics`` is the caller's turn-scoped dict,
    threaded onto that same object — the egress record from a [ESCALATE]
    cloud hop lands there, not in a dict of this function's own.

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
        return ChatResult(text=text)

    if not forwarded_query:
        # The model emitted the tag with nothing after it (or only
        # whitespace/a bare ":") — there is no forwarded query to
        # escalate, so neither door (answer_via_ha, answer_via_cloud) is
        # consulted.  Building an OutboundText from it and calling either
        # door would hand it an empty transcript — a caller-side
        # precondition an empty anonymize_turn call refuses on
        # (raises ValueError), not a cause either door's own refusal
        # vocabulary carries.
        return _pre_escalation_result(response)

    forwarded_is_personal = is_self_referential(
        forwarded_query,
        personal_referent_config=config.personal_referent,
    )

    forwarded = OutboundText(
        forwarded_query,
        config,
        diagnostics=diagnostics,
        history=history,
        model=model,
        tokenizer=tokenizer,
        speaker=speaker,
        speaker_id=speaker_id,
        language=language,
        is_personal=is_personal or forwarded_is_personal,
    )

    intent_label = intent.value if intent is not None else "unknown"
    if forwarded_is_personal:
        logger.info(
            "[ESCALATE] → HA door suppressed (intent=%s): forwarded query is personal",
            intent_label,
        )
    else:
        logger.info("[ESCALATE] → HA door (intent=%s): %s", intent_label, forwarded_query[:100])
        result = answer_via_ha(forwarded, ha_client)
        if result is not None:
            return result
    cloud_result = answer_via_cloud(forwarded, cloud_agent)
    if cloud_result is not None:
        logger.info(
            "[ESCALATE] → cloud fallback (intent=%s): %s", intent_label, forwarded_query[:100]
        )
        return cloud_result

    # All escalation paths exhausted — return pre-escalation text from local model
    return _pre_escalation_result(response)


def _build_messages(
    text: str,
    history: list[dict] | None,
    system_prompt: str,
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

    return messages
