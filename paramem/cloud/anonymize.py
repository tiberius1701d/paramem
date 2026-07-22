"""Anonymize a fact list / transcript for cloud egress — step (A) of the
cloud round trip.

Serves every cloud-bound path (session-tier extraction, graph-tier
enrichment, chat egress, and their calibration harnesses) through the SAME
two functions: :func:`anonymize_transcript` (the local-model call) and
:func:`anonymize` (the chain around it — normalize, reconcile, guard,
build the mapping).

Interface narrowing (2026-07-21): both functions take ``facts: list[dict]``
— never a ``SessionGraph`` or ``Relation``. A ``SessionGraph`` was a
CARRIER on this boundary, not an artifact: the pre-narrowing
``anonymize_for_cloud(graph, ...)`` touched ``graph`` only to (a) render
``graph.relations`` into the prompt's ``facts_json`` and (b) harvest
subject/object surfaces for the identity-reconciliation guard — both are
plain projections of ``Iterable[Relation]`` a caller can render once,
caller-side, in ``paramem/graph/``. The chat-egress caller had no graph at
all and manufactured an ephemeral one purely to satisfy the old signature
— this module no longer asks it to.

Likewise neither function loads its own prompt file: this package must
import nothing from ``paramem.graph`` (see the package docstring), and the
prompt-loading + calibration-override + provenance-recording chokepoint
(:func:`~paramem.graph.prompts._load_prompt`) lives there. The caller
(inside ``paramem/graph/``, already holding the active ``phase_trace``
scope this call's provenance records onto) loads
``anonymization.txt``/``anonymization_system.txt`` and passes the rendered
template strings in as ``user_prompt_template``/``system_prompt`` — this
is data, not a loader capability, so the layering holds without an
injected callable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from paramem.cloud.deanonymize import _extract_json_block
from paramem.cloud.placeholders import (
    _build_anonymization_mapping,
    _declared_placeholder_tokens,
    _normalize_anonymization_mapping,
)
from paramem.evaluation.recall import generate_answer
from paramem.models.loader import adapt_messages
from paramem.utils.identity import canonical, is_speaker_id
from paramem.utils.vram_guard import vram_scope

logger = logging.getLogger(__name__)

# Governs the anonymizer's local generate() call. 2048: the anonymization
# response is a mapping plus a rewritten transcript, both bounded by input
# size — smaller than the enrichment/plausibility budget.
_DEFAULT_ANONYMIZER_MAX_TOKENS = 2048
_DEFAULT_ANONYMIZER_TEMPERATURE = 0.0


def anonymize_transcript(
    facts: list[dict],
    model,
    tokenizer,
    *,
    scrub: set[str] | frozenset[str],
    transcript: str = "",
    max_tokens: int = _DEFAULT_ANONYMIZER_MAX_TOKENS,
    temperature: float = _DEFAULT_ANONYMIZER_TEMPERATURE,
    seed: int | None = None,
    user_prompt_template: str,
    system_prompt: str,
) -> tuple[dict | None, str, str]:
    """Identify the ``real_name -> placeholder`` mapping AND the
    model-authored anonymized transcript using the local model.

    Returns ``(mapping, anonymized_transcript, raw_output)``.  The model
    is shown ``facts`` (already rendered by the caller from its own
    relations/triples), ``transcript``, and ``scrub`` (the operator's
    PII-vocabulary hints, rendered into the prompt's ``{scrub_categories}``
    slot) and is the SOLE scope authority (no code-side entity-type gate):
    it decides which real values — names AND structured values (phone,
    email, …), regardless of owning entity type — are in scope, and
    returns BOTH the mapping AND its own rewrite of ``transcript`` with
    those values placeholdered (``anonymized_transcript``).  The CALLER
    still builds the anonymized FACT array deterministically from its own
    facts and ``mapping`` (see :func:`anonymize` and
    :func:`~paramem.cloud.placeholders.insert_placeholders`) — facts are
    never taken from the model's response — so a fact can never be lost,
    reworded, or dropped by the anonymizer, and a placeholder can never be
    glued into a predicate at this stage.

    ``mapping`` is ``None`` — the fail-closed signal — on PARSE
    FAILURE (the response was not a well-formed ``{"mapping": {...},
    "anonymized_transcript": [...]}`` envelope) OR when
    ``anonymized_transcript`` is missing/empty/whitespace-only.  Callers
    MUST never fall back to the original real-name transcript on this
    signal.  ``mapping`` is ``{}`` (with a non-empty
    ``anonymized_transcript``) when the model found nothing in scope to
    anonymize — a legitimate empty result, distinguishable from
    fail-closed.

    ``anonymized_transcript`` in the model's response is a JSON array of
    turn strings (one element per turn) per the ``configs/prompts/
    anonymization.txt`` contract — this keeps every turn on its own line
    so a multi-turn rewrite can never contain a literal newline inside a
    JSON string value (illegal per RFC 8259, the root cause of an
    observed multi-turn parse failure).  The array is joined with
    ``"\\n"`` back into a single transcript string before being returned.
    A plain ``str`` value is also still accepted and returned unchanged,
    since the model may emit the pre-contract shape; either way an empty
    result, a non-list/non-str value, or a list containing a non-``str``
    element is fail-closed.

    The raw model output is the third element so the calibration phase
    trace can record it without re-running the call.  ``anonymized_transcript``
    is ``""`` whenever ``mapping`` is ``None`` (the two fail-closed
    together).

    ``scrub`` is required — no implicit default.  An omitting caller would
    silently anonymize against a hidden default on a security-critical
    egress path; the single declared default is
    ``SanitizationConfig.scrub`` (``paramem/server/config.py``), which
    every caller of this function threads down explicitly.

    ``seed`` is forwarded verbatim to :func:`~paramem.evaluation.recall.
    generate_answer`.  At the default ``temperature=0.0`` (greedy
    decoding) it is a strict no-op.

    ``user_prompt_template``/``system_prompt`` are the ALREADY-LOADED
    prompt contents — the caller resolves
    ``configs/prompts/anonymization.txt`` /
    ``anonymization_system.txt`` (with calibration overrides and
    provenance recording) via
    :func:`~paramem.graph.prompts._load_prompt` and passes the text in;
    this function never touches the filesystem.

    THE only production caller of this function is :func:`anonymize`'s
    step 2 — every path that anonymizes for cloud egress reaches the
    model through the one chain, never this function directly.
    """
    prompt = user_prompt_template.format(
        scrub_categories=", ".join(sorted(scrub)),
        facts_json=json.dumps(facts, indent=2),
        transcript=transcript or "(no transcript provided)",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )

    # vram_scope: anonymization is the second-largest local generate after
    # main extraction and immediately precedes the plausibility filter.
    # Empty cache between this and the next phase so the filter's prefill
    # does not stack on top of anonymization's KV cache. Symmetric with
    # the other wraps.
    with vram_scope("anonymize"):
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
    logger.debug("Anonymization raw: %s", raw[:500])

    try:
        json_str = _extract_json_block(raw)
        data = json.loads(json_str)
        if not isinstance(data, dict) or "mapping" not in data:
            logger.warning("Anonymization returned unexpected format")
            return None, "", raw
        anon_transcript = data.get("anonymized_transcript")
        if isinstance(anon_transcript, list):
            # Contract shape: one turn string per element (see
            # configs/prompts/anonymization.txt) — join back into a
            # single transcript.  Any non-str element makes the whole
            # response untrustworthy, not just that element.
            if not anon_transcript or not all(isinstance(t, str) for t in anon_transcript):
                logger.warning("Anonymization returned a malformed anonymized_transcript array")
                return None, "", raw
            anon_transcript = "\n".join(anon_transcript)
        if not isinstance(anon_transcript, str) or not anon_transcript.strip():
            # Fail-closed: a mapping with no accompanying transcript
            # rewrite is not a safe egress artifact — never fall back to
            # the original real-name transcript.
            logger.warning("Anonymization returned missing/empty anonymized_transcript")
            return None, "", raw
        raw_mapping = data["mapping"]
        if not isinstance(raw_mapping, dict):
            logger.warning("Anonymization 'mapping' is not a dict")
            return None, "", raw
        return raw_mapping, anon_transcript, raw
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Anonymization parse failed: %s", e)
        return None, "", raw


@dataclass(frozen=True)
class AnonymizedContract:
    """Result of :func:`anonymize` — the ONE fail-closed vocabulary for
    every cloud-egress anonymize call.

    ``status``:

    * ``"opted_out"`` — ``scrub`` was empty (operator opt-out).  No model
      call was made.  ``anon_transcript`` is ``transcript`` verbatim
      (sourced from the argument, never a model artifact); ``forward`` /
      ``reverse`` are ``{}``.  A caller deriving the anonymized fact array
      via :func:`~paramem.cloud.placeholders.insert_placeholders` from its
      own facts and this empty ``forward`` gets the identity substitution
      (no-op).
    * ``"failed"`` — fail-closed: the local anonymizer failed to parse or
      returned a missing/empty ``anonymized_transcript`` (step 3), OR the
      domain-scoped privacy guard fired (step 6).  ``forward`` /
      ``reverse`` are empty; ``anon_transcript`` is
      ``""``.  Callers must NEVER fall back to the original real-name
      transcript on this status, and must NOT derive an anonymized fact
      array from this contract.  ``failure`` distinguishes which of the
      two causes fired — see below.
    * ``"ok"`` — the anonymizer ran (and, when ``identity_domain`` was
      supplied, reconciliation and the privacy guard both passed).
      ``forward`` / ``reverse`` may still be empty — a legitimate verdict
      ("ran, found nothing in scope"), not a failure; egress proceeds.

    ``failure`` — ``None`` except when ``status == "failed"``, where it is
    always one of:

    * ``"parse"`` — the local anonymizer call itself failed: a JSON parse
      failure, or a missing/empty ``anonymized_transcript`` (step 3).  No
      mapping was produced at all.
    * ``"guard"`` — the domain-scoped fail-closed guard fired (step 6):
      the anonymizer named something, but nothing survived to the final
      table — whether dropped by the model's own placeholder-shape
      validation (:func:`~paramem.cloud.placeholders.
      _normalize_anonymization_mapping`) or by the node-key reconciliation
      above.  This is a classification/identity-match failure, not a
      parse failure — the anonymizer ran and returned a well-formed
      mapping; it just didn't survive onto this chunk's actual domain.

    This field exists because ``status == "failed"`` alone collapses two
    causes a caller may need to react to differently (see
    :func:`~paramem.training.graph_enrich.enrich_graph`'s
    diagnostics), and ``rekey_dropped`` — a count, not a cause flag — is a
    lossy proxy: it is ``0`` both when the anonymizer never ran
    (``failure="parse"``) AND when it ran but every entry was dropped by
    shape validation before the reconciliation loop that increments it
    ever executed (``failure="guard"``).  Never reconstruct the cause from
    ``rekey_dropped``; read ``failure`` directly.

    ``reverse`` is the de-anonymization key: it must NEVER egress. The
    field that DOES egress is ``forward`` (used to placeholder outbound
    facts/transcript) plus ``anon_transcript``/``declared`` — naming this
    type ``AnonymizedContract`` rather than ``...Payload`` makes that
    asymmetry a property of the type, not just a comment on one field.
    """

    status: Literal["ok", "opted_out", "failed"]
    forward: dict[str, str]
    reverse: dict[str, str]
    anon_transcript: str
    declared: frozenset[str]
    norm_stats: dict[str, int]
    rekey_dropped: int
    raw: str
    failure: Literal["parse", "guard"] | None = None


def opted_out_contract(transcript: str) -> AnonymizedContract:
    """The ``status="opted_out"`` shape — the operator-opt-out (``scrub``
    empty) result, with ``transcript`` egressing verbatim.

    THE single constructor for this shape.  :func:`anonymize`'s own
    ``scrub``-empty branch uses it, and so does
    :func:`~paramem.graph.flows.anonymize_turn` — which needs to produce
    this exact contract WITHOUT calling :func:`anonymize` at all (its
    early ``if not scrub:`` return short-circuits BEFORE the local
    extraction pass that would otherwise anchor the anonymizer, so no
    compute is wasted anchoring a call that will never run; see that
    function's docstring). Before this helper existed the two sites
    independently hand-typed the same eight-field dataclass literal — a
    drift risk this collapses to one.
    """
    return AnonymizedContract(
        status="opted_out",
        forward={},
        reverse={},
        anon_transcript=transcript,
        declared=frozenset(),
        norm_stats={"inverted": 0, "dropped": 0},
        rekey_dropped=0,
        raw="",
        failure=None,
    )


def anonymize(
    facts: list[dict],
    model,
    tokenizer,
    *,
    transcript: str,
    scrub: set[str] | frozenset[str],
    speaker_name: str | None = None,
    identity_domain: Iterable[str] | None = None,
    max_tokens: int = _DEFAULT_ANONYMIZER_MAX_TOKENS,
    seed: int | None = None,
    user_prompt_template: str,
    system_prompt: str,
) -> AnonymizedContract:
    """Anonymize a fact list + transcript for cloud egress — THE one
    anonymize chain every cloud-bound path (session-tier extraction,
    graph-tier enrichment, chat egress, and their calibration harnesses)
    composes through.

    Serves both "facts but no transcript" (graph tier: ``transcript=""``)
    and "transcript but no facts" (chat egress: ``facts=[]``) via the SAME
    signature — no flag, no branch.  ``facts`` may be empty (the caller's
    own :func:`~paramem.cloud.placeholders.insert_placeholders` call then
    gets ``[]``); ``transcript`` may be empty (``anon_transcript`` comes
    back ``""`` on the opt-out path, sourced from the argument).

    Fixed sequence:

    1. ``scrub`` empty -> ``status="opted_out"``, empty tables,
       ``anon_transcript=transcript`` verbatim.  A caller deriving the
       anonymized fact array gets the identity (empty-mapping)
       substitution.
    2. :func:`anonymize_transcript` -> ``(llm_mapping,
       model_anon_transcript, raw)``.
    3. ``llm_mapping is None`` -> ``status="failed"``, empty tables,
       ``anon_transcript=""``.  Never falls back to the real-name
       transcript.
    4. :func:`~paramem.cloud.placeholders._normalize_anonymization_mapping`
       — the ONE normalize call; ``norm_stats`` is a LIVE signal callers
       persist (``{"inverted": N, "dropped": N}``).
    5. **Identity reconciliation** (only when ``identity_domain is not
       None`` — the graph tier's pre-step, generalized as data).  Every
       mapping key is folded through :func:`~paramem.utils.identity.
       canonical` and matched against the (also-canonicalized) entries of
       ``identity_domain``.  A unique match is re-keyed onto the domain
       surface with the model's placeholder preserved verbatim; a miss or
       an ambiguous multi-match is dropped and counted into
       ``rekey_dropped``.  ``identity_domain=None`` (session tier / chat
       egress / calibration) skips this step entirely — those mapping
       keys are free-text surfaces with no closed domain to reconcile
       against.
    6. **Domain-scoped fail-closed guard** — fires ONLY when
       ``identity_domain is not None`` AND the reconciled-from mapping was
       non-empty AND the reconciled mapping came back empty AND ``facts``'
       subject/object endpoints contain a non-speaker name ->
       ``status="failed"``.  The guard domain is deliberately derived from
       ``facts`` — NOT from ``identity_domain`` — because the two differ:
       ``identity_domain`` (the rekey domain) is a caller-supplied node
       list that may be trimmed to a size cap; the guard domain is the
       facts' actual subject/object endpoints, a subset of that list once
       trimming drops nodes whose only surviving edges fell outside the
       chunk.  Fusing the two would silently reject a chunk whose
       surviving edges are all speaker-only just because its (larger,
       untrimmed) node list still lists a non-speaker node — a
       false-positive regression, not a privacy fix.  This coupling —
       ``identity_domain`` gates BOTH step 5 and step 6 — is why a caller
       passing a domain must read this docstring: an
       ``identity_domain=None`` case never reconciles and never guards.
    7. :func:`~paramem.cloud.placeholders._build_anonymization_mapping`
       (``speaker_name=speaker_name``) -> ``(forward, reverse)``.  THE
       speaker-value guard now applies on every path by construction — no
       code path is left that inverts an unfiltered forward map.
    8. :func:`~paramem.cloud.placeholders._declared_placeholder_tokens`
       (``reverse``) -> ``declared`` — the CORE declared vocabulary (no
       Cloud bindings yet at this stage; see
       :meth:`~paramem.cloud.deanonymize.CloudScope.response` for the
       union used at deanon time).

    This function does NOT build the anonymized fact array itself — every
    production reader derives it on demand instead, via
    :func:`~paramem.cloud.placeholders.insert_placeholders` over its own
    facts and this contract's ``forward`` map: the ``enrich`` stage
    (:mod:`paramem.graph.stage_enrich`), ``request_graph_enrichment``
    (:mod:`paramem.graph.extractor`), and the ``/calibrate/anonymize``
    handler (:mod:`paramem.server.calibrate`, status-gated to ``[]`` on
    ``status == "failed"``).

    ``user_prompt_template``/``system_prompt`` are the caller-loaded
    ``anonymization.txt``/``anonymization_system.txt`` contents — see
    :func:`anonymize_transcript`'s docstring for why this package never
    loads a prompt file itself.
    """
    if not scrub:
        return opted_out_contract(transcript)

    llm_mapping, model_anon_transcript, raw = anonymize_transcript(
        facts,
        model,
        tokenizer,
        transcript=transcript,
        scrub=scrub,
        max_tokens=max_tokens,
        seed=seed,
        user_prompt_template=user_prompt_template,
        system_prompt=system_prompt,
    )
    if llm_mapping is None:
        return AnonymizedContract(
            status="failed",
            forward={},
            reverse={},
            anon_transcript="",
            declared=frozenset(),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw=raw or "",
            failure="parse",
        )

    mapping, norm_stats = _normalize_anonymization_mapping(llm_mapping)

    rekey_dropped = 0
    if identity_domain is not None:
        domain = list(identity_domain)
        canon_to_domain: dict[str, str] = {}
        ambiguous_canon: set[str] = set()
        for d in domain:
            c = canonical(str(d))
            if c in canon_to_domain and canon_to_domain[c] != d:
                # Two distinct domain entries canonicalizing identically —
                # fail closed on this entry rather than silently pick one.
                ambiguous_canon.add(c)
            else:
                canon_to_domain[c] = d

        # Derived from the RAW model output (``llm_mapping``), not the
        # post-normalize ``mapping`` — the guard below must fire whenever
        # the model named something and NOTHING survived to the final
        # table, regardless of WHICH stage (normalize's shape check OR
        # this reconciliation step) is what dropped it.  Deriving this
        # from ``mapping`` instead would make every entry
        # ``_normalize_anonymization_mapping`` drops (e.g. a 7B emitting
        # consistently lowercase placeholders, which fails
        # PLACEHOLDER_SHAPE_RE on both sides) invisible to the guard: the
        # chunk would egress its real names verbatim instead of failing
        # closed with ``failure="guard"`` — the caller reads that field,
        # not ``rekey_dropped`` (which stays ``0`` in the
        # shape-validation-drop case, since the reconciliation loop below
        # never runs on an entry normalize already removed).
        reconciled_from_nonempty = bool(llm_mapping)
        reconciled: dict[str, str] = {}
        for real, placeholder in mapping.items():
            c = canonical(real)
            if c in ambiguous_canon or c not in canon_to_domain:
                rekey_dropped += 1
                continue
            reconciled[canon_to_domain[c]] = placeholder
        mapping = reconciled

        endpoint_names = {str(f.get(field, "")) for f in facts for field in ("subject", "object")}
        endpoint_names = {n for n in endpoint_names if n and not is_speaker_id(n)}
        if reconciled_from_nonempty and endpoint_names and not mapping:
            return AnonymizedContract(
                status="failed",
                forward={},
                reverse={},
                anon_transcript="",
                declared=frozenset(),
                norm_stats=norm_stats,
                rekey_dropped=rekey_dropped,
                raw=raw or "",
                failure="guard",
            )

    forward, reverse = _build_anonymization_mapping(mapping, speaker_name=speaker_name)
    declared = frozenset(_declared_placeholder_tokens(reverse))

    return AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript=model_anon_transcript,
        declared=declared,
        norm_stats=norm_stats,
        rekey_dropped=rekey_dropped,
        raw=raw or "",
        failure=None,
    )
