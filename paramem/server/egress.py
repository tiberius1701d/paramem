"""External egress — the one primitive both answering legs (HA and cloud)
send outbound text through.

The two external legs carry different policies, because they cross
different trust boundaries:

* **Cloud leg** (:func:`answer_via_cloud`) — a third party's servers.
  Governed by ``sanitization.cloud_mode`` exactly as configured
  (``block`` / ``anonymize`` / ``both``): personal-value substitution
  through the anonymize chain, a policy-gated personal-query refusal, and
  a restore of the reply against the same contract.
* **HA leg** (:func:`answer_via_ha`) — the configured Home Assistant
  conversation agent. The turn goes out and the reply comes back exactly
  as they are: no anonymize contract is built, no substitution runs, no
  restore runs, and the leg is never closed by a personal verdict. Which
  agent handles the turn, local or hosted, is the operator's choice; the
  security documentation carries the recommendation.

:class:`OutboundText` holds one outbound text, the inputs the anonymize
chain needs, and the single :class:`~paramem.cloud.anonymize.AnonymizedContract`
built for it — memoised, so the anonymize chain runs at most once per
object even when the cloud leg is read from it more than once (a leg
followed by a caller-side fallback). ``cloud_agent`` / ``cloud_permitted``
/ ``ha_client`` are the doors' own parameters, never fields of the shared
object.

This module owns everything that decides what leaves the house:
:class:`OutboundText`, :func:`answer_via_ha`, :func:`answer_via_cloud`,
:func:`_escalate_to_cloud` (the cloud transport primitive, one caller),
:func:`_sanitize_history`, :func:`_stamp_leg`, :func:`_refuse_failed_contract`,
and the per-leg record vocabularies (:data:`LEG_NAMES` / :data:`_EGRESS_VALUES`
/ :data:`_REFUSAL_VALUES`). ``MAX_HISTORY_TURNS`` belongs to
:mod:`paramem.server.session_buffer` instead — the property it bounds is
the read of that buffer's own history, not an egress concern — and this
module imports it from there.
:mod:`paramem.server.inference` keeps local routing — the personal probe,
the base-model reasoning generate, the escalation-tag detection — and
imports the two doors from here. The boundary is "what leaves the house"
against "what the local model does", and the import direction is one-way:
``inference`` → ``egress`` → ``chat_result``. Nothing in this module
imports from :mod:`paramem.server.inference`.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from paramem.cloud.providers.base import CloudAgent
from paramem.server.chat_result import ChatResult
from paramem.server.config import ServerConfig
from paramem.server.prompts import cloud_serving_system_prompt, language_instruction
from paramem.server.sanitizer import is_self_referential
from paramem.server.session_buffer import MAX_HISTORY_TURNS
from paramem.server.tools.ha_client import HAClient

if TYPE_CHECKING:
    from paramem.cloud.anonymize import AnonymizedContract

logger = logging.getLogger(__name__)

# Per-leg record vocabulary — the closed set of legs and the closed sets of
# egress/refusal values every stamp onto a turn's diagnostics dict is
# checked against.  Public (no leading underscore): both
# ``inference._leg_open`` and ``app._resolve_probe_route`` validate a
# forced-leg selection against this same tuple rather than a bare literal.
LEG_NAMES = ("cloud", "ha")
_EGRESS_VALUES = ("scrubbed", "verbatim")
_REFUSAL_VALUES = (
    "not_permitted",
    "personal_blocked",
    "guard",
    "model_unavailable",
    "scan_failed",
    "unresolved_placeholder",
)


def _sanitize_history(history: list[dict] | None) -> list[dict]:
    """Drop-gate conversation history for an external leg: self-referential
    turns are removed.

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
            logger.info("Dropped self-referential history turn from an outbound payload")
            continue
        sanitized.append({"role": role, "text": text})
    return sanitized


@dataclass
class OutboundText:
    """One outbound text and the scrub state every external leg shares for it.

    Built once per text a leg may send: the turn (with its history) on the
    routed legs, or the model-authored forwarded query behind
    ``[ESCALATE]``, which is a different artifact and gets its own object.
    The anonymize chain runs at most once per object, lazily — the HA leg
    never calls :meth:`contract`, so an HA-miss -> cloud-fallback turn
    builds the contract at most once, on the cloud leg's own first read.

    ``reverse`` lives on the held contract and never leaves this object:
    the cloud leg exposes only the substituted outbound surfaces and the
    restored reply; the HA leg exposes ``self.text`` unchanged.

    Attributes:
        text: The text this object may send — a turn or a forwarded query.
        config: The live server config.
        diagnostics: The caller's turn-scoped diagnostics dict; both doors
            stamp their leg's outcome onto it via :func:`_stamp_leg`.
        history: Prior conversation turns, ungated, or ``None``.
        model: The resident base model, or ``None`` on a cloud-only
            deferral.
        tokenizer: Paired with *model*.
        speaker: The speaker's display name, or a raw ``speaker{N}`` token
            for an undisclosed anonymous speaker, or ``None``.
        speaker_id: The speaker's canonical ``speaker{N}`` token, or
            ``None``.
        language: BCP-47 language code, or ``None``.
        is_personal: The turn's (or forwarded query's) personal verdict.
    """

    text: str
    config: ServerConfig
    diagnostics: dict[str, Any]
    history: list[dict] | None = None
    model: Any = None
    tokenizer: Any = None
    speaker: str | None = None
    speaker_id: str | None = None
    language: str | None = None
    is_personal: bool = False

    _contract: "AnonymizedContract | None" = field(default=None, init=False, repr=False)
    _gated_history: list[dict] | None = field(default=None, init=False, repr=False)

    def gated_history(self) -> list[dict]:
        """The drop-gated history turns, computed once and memoised.

        Returns:
            ``_sanitize_history(self.history)``, the same list on every
            call.
        """
        if self._gated_history is None:
            self._gated_history = _sanitize_history(self.history)
        return self._gated_history

    def contract(self) -> "AnonymizedContract":
        """The anonymize chain's result for this object's text, memoised.

        Calls :func:`~paramem.graph.flows.anonymize_turn` at most once per
        object. A raise propagates — no handler here.

        Returns:
            The memoised :class:`~paramem.cloud.anonymize.AnonymizedContract`.
        """
        if self._contract is None:
            from paramem.graph.flows import anonymize_turn

            self._contract = anonymize_turn(
                self.text,
                self.model,
                self.tokenizer,
                history=self.gated_history(),
                speaker_id=self.speaker_id,
                speaker_name=self.speaker,
                categories=self.config.sanitization.scrub_categories,
                token_envelope=self.config.consolidation.extraction_anonymize_token_envelope,
            )
        return self._contract

    def outbound_text(self) -> str:
        """The current turn's text, substituted over this object's forward table.

        Cloud only — the HA leg sends ``self.text`` unchanged and never
        calls this method.

        Returns:
            *self.text* with every scrubbed value replaced by its
            placeholder.
        """
        from paramem.cloud.placeholders import _substitute_whole_words

        contract = self.contract()
        return _substitute_whole_words(self.text, contract.forward)

    def outbound_history(self) -> list[dict]:
        """The drop-gated history turns, substituted over this object's forward table.

        Cloud only — the HA leg sends no history.

        Returns:
            ``gated_history()`` with each turn's ``text`` substituted over
            ``contract().forward``.
        """
        from paramem.cloud.placeholders import _substitute_whole_words

        forward = self.contract().forward
        return [
            {**turn, "text": _substitute_whole_words(turn["text"], forward)}
            for turn in self.gated_history()
        ]

    def restore(self, reply: str, *, sent: tuple[str, ...]) -> str | None:
        """De-anonymize *reply* against this object's contract.

        Args:
            reply: The external leg's raw reply text.
            sent: Every string the recipient was actually shown (the
                outbound text, plus every outbound history turn for the
                cloud leg) — scopes ``observed`` to what was actually sent.

        Returns:
            The de-anonymized reply, or ``None`` when a declared-but-
            unobserved (or otherwise unresolved) placeholder survives —
            the caller must refuse the whole reply on ``None``.
        """
        from paramem.cloud.deanonymize import CloudScope, deanonymize_text

        scope = CloudScope.response(self.contract(), cloud_bindings=None, sent=sent)
        return deanonymize_text(scope, reply)


def _stamp_leg(
    diagnostics: dict[str, Any],
    leg: str,
    *,
    egress: str | None = None,
    refusal: str | None = None,
) -> None:
    """Write one leg's outcome into *diagnostics*, popping the leg's other key.

    ``{leg}_egress`` and ``{leg}_refusal`` describe the same turn's single
    outcome for that leg and must never both be present — a leg's dict can
    be handed to a door more than once for the same turn (e.g. a refusal
    followed by a later success on the same leg), and the last outcome is
    the record.  The two legs' keys are independent: both may be present
    on one turn's dict (e.g. HA answered nothing and cloud answered).

    Args:
        diagnostics: The turn-scoped diagnostics dict to stamp.
        leg: One of :data:`LEG_NAMES`.
        egress: One of :data:`_EGRESS_VALUES`, or ``None``.
        refusal: One of :data:`_REFUSAL_VALUES`, or ``None``.

    Raises:
        ValueError: If *leg* is not in :data:`LEG_NAMES`; if neither or
            both of *egress*/*refusal* are given; or if the given value is
            not in its closed vocabulary.
    """
    if leg not in LEG_NAMES:
        raise ValueError(f"_stamp_leg: unknown leg {leg!r}; expected one of {LEG_NAMES}")
    if (egress is None) == (refusal is None):
        raise ValueError("_stamp_leg: exactly one of egress or refusal must be given")
    if egress is not None:
        if egress not in _EGRESS_VALUES:
            raise ValueError(
                f"_stamp_leg: unknown egress value {egress!r}; expected one of {_EGRESS_VALUES}"
            )
        diagnostics.pop(f"{leg}_refusal", None)
        diagnostics[f"{leg}_egress"] = egress
    else:
        if refusal not in _REFUSAL_VALUES:
            raise ValueError(
                f"_stamp_leg: unknown refusal value {refusal!r}; expected one of {_REFUSAL_VALUES}"
            )
        diagnostics.pop(f"{leg}_egress", None)
        diagnostics[f"{leg}_refusal"] = refusal


def _refuse_failed_contract(outbound: OutboundText, leg: str) -> None:
    """Stamp *leg*'s refusal from a ``status="failed"`` contract.

    Maps the contract's three ``failure`` values one to one onto the
    cloud leg's refusal vocabulary — the HA leg never builds a contract
    and never calls this function.

    Args:
        outbound: The object whose ``contract().failure`` names the cause.
        leg: The leg refusing (``"cloud"`` in production).

    Raises:
        RuntimeError: If ``outbound.contract().failure`` is none of
            ``"guard"``, ``"model_unavailable"``, ``"scan_failed"`` — an
            invariant violation, not a cause to launder into an existing
            bucket.
    """
    failure = outbound.contract().failure
    if failure in ("guard", "model_unavailable", "scan_failed"):
        _stamp_leg(outbound.diagnostics, leg, refusal=failure)
    else:
        raise RuntimeError(
            f"_refuse_failed_contract: unrecognised failure value {failure!r} for leg "
            f"{leg!r} — expected 'guard', 'model_unavailable' or 'scan_failed'"
        )


def answer_via_ha(
    outbound: OutboundText,
    ha_client: HAClient | None,
) -> ChatResult | None:
    """Send *outbound* to the HA conversation agent — the HA egress door.

    The HA leg carries the turn verbatim on every path: no anonymize
    contract is built, no substitution runs, and no restore runs on the
    reply. It sends the turn to whichever HA conversation agent the
    operator has configured; the choice of that agent, local or hosted,
    is the operator's, and the security documentation carries the
    recommendation. This leg is never closed by a personal verdict
    either: HA reachability does not depend on ``outbound.is_personal``
    at all.

    Sequence:

    1. ``ha_client is None`` or no ``ha_agent_id`` configured -> ``None``,
       nothing sent, nothing stamped.
    2. Stamp the leg's egress at the send boundary — always ``"verbatim"``.
    3. Call the HA client's ``conversation_process`` with ``outbound.text``
       unchanged.
    4. ``None`` reply -> return ``None`` (the caller owns the fall-through
       to the next mechanism in its chain).
    5. Return the reply exactly as it came, wrapped in a
       :class:`~paramem.server.chat_result.ChatResult`.

    Args:
        outbound: The text to send.
        ha_client: The HA client, or ``None`` when HA is not configured.

    Returns:
        The HA reply as a :class:`~paramem.server.chat_result.ChatResult`,
        or ``None`` when this leg did not answer.
    """
    if ha_client is None or not outbound.config.ha_agent_id:
        return None

    _stamp_leg(outbound.diagnostics, "ha", egress="verbatim")

    reply = ha_client.conversation_process(
        outbound.text,
        agent_id=outbound.config.ha_agent_id,
        language=outbound.language,
        supported_languages=outbound.config.tools.ha.supported_languages,
    )
    if reply is None:
        return None

    return ChatResult(text=reply, escalated=True)


def answer_via_cloud(
    outbound: OutboundText,
    cloud_agent: CloudAgent | None,
    *,
    cloud_permitted: bool = True,
) -> ChatResult | None:
    """Apply the configured cloud-egress policy and call cloud accordingly.

    The sole cloud-egress door: every caller — local-mode routing
    (:mod:`paramem.server.inference`) and cloud-only routing
    (``paramem.server.app._relay_route`` and the probe door's forced
    ``route=cloud*`` selection) — reaches :func:`_escalate_to_cloud` only
    through here, and this is the only site that reads
    ``config.sanitization.cloud_mode``.

    +-------------+----------------------+----------------------+
    | mode        | PERSONAL query       | non-PERSONAL query   |
    +=============+======================+======================+
    | ``block``   | None (blocked)       | cloud verbatim        |
    | ``anonymize`` | anon -> cloud -> deanon | anon -> cloud -> deanon |
    | ``both``    | None (blocked)       | anon -> cloud -> deanon |
    +-------------+----------------------+----------------------+

    Per-query safety: when an anonymizing path is selected and the local
    anonymizer fails to produce a mapping (the base model was not resident,
    the scan call failed, or the domain-scoped fail-closed guard fired),
    this call returns ``None`` so the caller falls back without sending
    anything to the cloud.

    ``cloud_permitted`` defaults to ``True``: on the local-mode path, cloud
    egress is already gated upstream by ``cloud_agent`` presence, so local
    callers never compute it. Cloud-only callers compute it from
    ``_state["cloud_only_reason"]`` and ``config.cloud.allow_degraded_serving``
    and thread the result in explicitly.

    Egress record: every returned :class:`~paramem.server.chat_result.ChatResult`
    and every ``None`` this function returns while ``cloud_agent`` was
    present is stamped into ``outbound.diagnostics`` via :func:`_stamp_leg`
    — see that function's docstring for the key-presence contract.

    Args:
        outbound: The text to send.
        cloud_agent: The cloud agent, or ``None`` when cloud is not
            configured.
        cloud_permitted: Whether the cloud leg may be used at all this
            call.

    Returns:
        The cloud result on success, or ``None`` when policy or per-query
        safety blocks the call.
    """
    if cloud_agent is None:
        return None

    if not cloud_permitted:
        _stamp_leg(outbound.diagnostics, "cloud", refusal="not_permitted")
        logger.warning(
            "Cloud escalation blocked: degraded serving not permitted "
            "(set cloud.allow_degraded_serving: true to enable)"
        )
        return None

    config = outbound.config
    cloud_mode = config.sanitization.cloud_mode
    if cloud_mode not in {"block", "anonymize", "both"}:
        # Unknown / mock value — fall back to the safest mode (block).
        # Production paths can't reach this branch because
        # SanitizationConfig.__post_init__ validates the field; this guard
        # protects test mocks and any future config drift.
        cloud_mode = "block"

    blocks_personal = cloud_mode in {"block", "both"}
    anonymizes_outbound = cloud_mode in {"anonymize", "both"}

    if outbound.is_personal and blocks_personal:
        _stamp_leg(outbound.diagnostics, "cloud", refusal="personal_blocked")
        return None

    if anonymizes_outbound:
        contract = outbound.contract()
        if contract.status == "failed":
            _refuse_failed_contract(outbound, "cloud")
            return None

        anon_text = outbound.outbound_text()
        sanitized_history = outbound.outbound_history()
        _stamp_leg(
            outbound.diagnostics, "cloud", egress="scrubbed" if contract.forward else "verbatim"
        )

        result = _escalate_to_cloud(
            anon_text,
            cloud_agent,
            config,
            sanitized_history=sanitized_history,
            language=outbound.language,
        )
        # "observed" means "tokens the recipient was actually shown" — and
        # history is shown: _escalate_to_cloud passes sanitized_history to
        # cloud_agent.call(history=...), which providers render verbatim
        # into the messages sent to the provider.  sent must include every
        # history turn's text alongside the current turn or a token
        # occurring only in a history turn is wrongly scoped out.
        restored = outbound.restore(
            result.text,
            sent=(anon_text, *(turn["text"] for turn in sanitized_history)),
        )
        if restored is None:
            # Fail-closed: a declared-but-unobserved placeholder (or
            # otherwise unresolved token) survived in the cloud's
            # response — never forward it with a residual placeholder.
            logger.warning("Cloud response carried an unresolved placeholder — blocking")
            _stamp_leg(outbound.diagnostics, "cloud", refusal="unresolved_placeholder")
            return None
        result.text = restored
        return result

    # cloud_mode=block + non-PERSONAL: current turn goes verbatim (the
    # personal verdict already cleared it).  History is still drop-gated —
    # an old turn can be personal even when this one is not.
    sanitized_history = outbound.gated_history()
    _stamp_leg(outbound.diagnostics, "cloud", egress="verbatim")
    return _escalate_to_cloud(
        outbound.text,
        cloud_agent,
        config,
        sanitized_history=sanitized_history,
        language=outbound.language,
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

    This is the transport primitive, not a door: it makes no policy
    decision of its own and has exactly one caller, :func:`answer_via_cloud`.

    The system prompt carries NO identity line at all — no ``speaker_id``
    token and no display name.  Unlike the local reasoning leg
    (``_build_speaker_prefix``, fed the ``speaker{N}`` token), the cloud
    system prompt is the bare
    :func:`~paramem.server.prompts.cloud_serving_system_prompt` plus only
    the language instruction.  This is scoped to the system prompt only:
    *sanitized_history* can still carry a ``speaker{N}`` token verbatim —
    an accepted posture, not a leak — so "cloud never learns who it is
    talking to" would overclaim.

    Args:
        text: The query text.  Already policy-processed by
            :func:`answer_via_cloud` before this call.
        cloud_agent: Cloud agent to delegate to.
        config: Server config.
        sanitized_history: Conversation history turns, ALREADY drop-gated
            (and, under an anonymizing ``cloud_mode``, placeholdered) by
            :func:`answer_via_cloud` — this function does not sanitize.
        language: BCP-47 language code.
    """
    sanitized_history = sanitized_history or []

    lang_instr = language_instruction(language, config)
    base = cloud_serving_system_prompt()
    prompt = (lang_instr + " " + base) if lang_instr else base

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
