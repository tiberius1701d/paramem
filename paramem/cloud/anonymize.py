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
``anonymization.txt``/``anonymization_system.txt`` (or, for the graph
tier's fact-only calls, ``anonymization_facts.txt``) and passes the
rendered template strings in as ``user_prompt_template``/``system_prompt``
— this is data, not a loader capability, so the layering holds without an
injected callable.

Fact-boundary slicing: a single ``anonymize()`` call may cost more than
one local ``generate()`` — see :func:`_slice_facts_to_envelope` and the
per-slice loop inside :func:`anonymize`.
A transcript is atomic (it is never split), so slicing
applies only when ``transcript`` is empty (the graph tier); the session
tier and chat egress always make exactly one local call.

Dynamic VRAM clamp (owner-approved 2026-07-28, live-fold evidence: a
packer-correct 8,192-token call still faulted "device not ready" at 1,191
MiB free): the configured ``token_envelope`` is the operator CEILING, not
a guarantee of what live free VRAM can support at call time — free VRAM
varies within one fold. :func:`anonymize` measures free VRAM ONCE at its
own entry, via :func:`~paramem.utils.vram_guard.effective_token_envelope`,
and threads the resulting effective (possibly smaller) envelope to both
the fact packer and every slice's :func:`anonymize_transcript` call — see
:func:`anonymize`'s docstring for why one entry-time measurement is
conservative-safe for later slices too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from paramem.cloud.deanonymize import _extract_json_block
from paramem.cloud.placeholders import (
    _build_anonymization_mapping,
    _declared_placeholder_tokens,
    _normalize_anonymization_mapping,
    mint_placeholder,
    placeholder_prefix,
)
from paramem.evaluation.recall import generate_answer
from paramem.models.loader import adapt_messages
from paramem.utils.identity import canonical, is_speaker_id
from paramem.utils.tokens import ANONYMIZE_ENVELOPE_TOKENS, estimate_tokens
from paramem.utils.vram_guard import effective_token_envelope, vram_scope

logger = logging.getLogger(__name__)

# Total tokens (prompt + output) one local anonymize call may occupy. 8192 is
# the sequence length vram.vram_cache_headroom_gib: 1.5 was booked against
# (configs/server.yaml.example — "single-sequence 8192-token Mistral KV cache
# is ~1 GiB; 0.5 GiB margin for activations"). One envelope, no second cap:
# every caller derives its output allowance from this single budget (output =
# envelope − prompt) — never a separate max_tokens knob. Reads
# paramem.utils.tokens.ANONYMIZE_ENVELOPE_TOKENS — the one executable home
# for the literal, since paramem.graph.document_chunker and
# paramem.server.session_buffer also need this value and must stay
# importable without the cloud package (see paramem/utils/tokens.py).
_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE: int = ANONYMIZE_ENVELOPE_TOKENS
# Structural floor: below this no well-formed
# {"mapping": ..., "anonymized_transcript": ...} envelope can complete, so a
# smaller allowance only guarantees a truncated (fail-closed) response.
_MIN_ANONYMIZER_OUTPUT_TOKENS: int = 256
# Per-mapping-entry output overhead: quotes(4) + ": "(2) + placeholder token
# (~3: Person/_/12) + ","(1) ≈ 10. A property of the response JSON schema,
# not a tunable — the real value's OWN token cost is measured separately
# (see _mapping_reserve_tokens) since it varies with the surface's length.
_MAPPING_ENTRY_OVERHEAD_TOKENS: int = 10
# The response's fixed JSON skeleton (`{"mapping": {...},
# "anonymized_transcript": [...]}`) plus slack.
_OUTPUT_JSON_ENVELOPE_TOKENS: int = 48
# Fixed per-fact array-separator overhead (comma between elements) applied
# by the O(n) slice packer below — a conservative constant, not a measured
# token count (the exact accounting happens where it binds, at the actual
# rendered prompt inside `anonymize_transcript`).
_FACT_SEPARATOR_TOKENS: int = 1
_DEFAULT_ANONYMIZER_TEMPERATURE = 0.0


def _render_anonymize_prompt(
    facts: list[dict],
    tokenizer,
    *,
    scrub: set[str] | frozenset[str],
    transcript: str,
    user_prompt_template: str,
    system_prompt: str,
    speaker_id: str | None = None,
    speaker_anchor_template: str = "",
) -> str:
    """THE prompt renderer: ``template.format(...)`` +
    ``tokenizer.apply_chat_template(...)``.

    One implementation, shared by :func:`anonymize_transcript`'s generate
    call and :func:`_slice_facts_to_envelope`'s overhead probe (the
    packer measures the SAME rendering the real call will produce, minus
    the per-fact content it hasn't decided on yet).

    ``user_prompt_template`` may be either the session-tier
    ``anonymization.txt`` (which has a ``{transcript}`` slot) or the
    graph-tier ``anonymization_facts.txt`` (which does not) — passing
    ``transcript=`` as a keyword argument to ``str.format`` is safe
    either way, since ``str.format`` ignores unused keyword arguments
    when the template has no matching ``{transcript}`` placeholder.

    ``speaker_id``/``speaker_anchor_template`` together fill
    ``anonymization.txt``'s ``{speaker_anchor_section}`` slot — the
    fold-onto-token speaker-anchor rule and its worked example, THIS
    session's own ``speaker{{N}}`` token telling the anonymizer LLM which
    placeholder a user's own real-name mention may fold onto. The section
    renders ONLY when ``speaker_id`` is given (truthy): a caller-loaded
    ``speaker_anchor_template`` (``configs/prompts/
    anonymization_speaker_anchor.txt``, itself formatted with
    ``speaker_id=speaker_id``) is inserted; otherwise the slot renders as
    ``""`` — no anchor rule, no worked example, and the surrounding prose
    reads cleanly either way (pinned by
    ``tests/test_prompts_contract.py``). ``speaker_id`` alone (with no
    template) also renders empty, matching every other caller that omits
    ``speaker_anchor_template`` (the graph tier, which never has a
    single-session speaker to name at all, and any calibration caller
    that omits both). Safe against a template with no matching slot
    either way, since ``str.format`` ignores unused keyword arguments
    (``anonymization_facts.txt`` has neither ``{speaker_id}`` nor
    ``{speaker_anchor_section}``).
    """
    speaker_anchor_section = (
        speaker_anchor_template.format(speaker_id=speaker_id)
        if speaker_id and speaker_anchor_template
        else ""
    )
    prompt = user_prompt_template.format(
        scrub_categories=", ".join(sorted(scrub)),
        facts_json=json.dumps(facts),
        transcript=transcript or "(no transcript provided)",
        speaker_id=speaker_id or "",
        speaker_anchor_section=speaker_anchor_section,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )


def _mapping_reserve_tokens(surface: str, tokenizer) -> int:
    """Output tokens one distinct real-value surface's mapping entry costs.

    The surface's own token cost (it appears as the JSON key, quoted)
    plus the fixed per-entry schema overhead
    (:data:`_MAPPING_ENTRY_OVERHEAD_TOKENS` — quotes, separator, the
    placeholder value, and the trailing comma, none of which vary with
    the surface's length).
    """
    return estimate_tokens(surface, tokenizer) + _MAPPING_ENTRY_OVERHEAD_TOKENS


def _output_reserve_tokens(facts: list[dict], tokenizer, *, transcript_tokens: int = 0) -> int:
    """Whole-call output reserve: the JSON skeleton, the transcript echo
    (the rewrite is a faithful copy of the input transcript), and one
    mapping entry per distinct endpoint surface.

    Composes :func:`_mapping_reserve_tokens` once per distinct
    ``subject``/``object`` surface across ``facts`` — never re-derives
    the per-entry cost.
    """
    surfaces = {str(f.get(field, "")) for f in facts for field in ("subject", "object")}
    surfaces = {s for s in surfaces if s}
    mapping_reserve = sum(_mapping_reserve_tokens(s, tokenizer) for s in surfaces)
    return _OUTPUT_JSON_ENVELOPE_TOKENS + transcript_tokens + mapping_reserve


def _slice_facts_to_envelope(
    facts: list[dict],
    tokenizer,
    *,
    scrub: set[str] | frozenset[str],
    user_prompt_template: str,
    system_prompt: str,
    token_envelope: int,
) -> list[list[dict]]:
    """Pack *facts* into slices whose rendered prompt plus derived output
    reserve fits *token_envelope*. Fact boundaries only — a fact is never
    split.

    O(n) tokenizations: the template skeleton is measured once
    (``base``), then each fact's own JSON rendering is measured once as
    it is considered for the current slice — never a full slice
    re-render per fact.

    1. ``base`` = the rendered empty-facts prompt's token count — the
       template skeleton that dominates a small chunk.
    2. Per fact: its own JSON fragment's token count, plus the mapping
       reserve for any of its ``subject``/``object`` surfaces not
       already counted in the current slice.
    3. A fact is accepted into the current slice while
       ``base + slice_fragment_tokens + separators +
       max(_MIN_ANONYMIZER_OUTPUT_TOKENS, _OUTPUT_JSON_ENVELOPE_TOKENS +
       slice_reserve) <= token_envelope``; otherwise the current slice
       closes and a new one starts with this fact.  The floor mirrors
       :func:`anonymize_transcript`'s own ``desired_reserve = max(
       _MIN_ANONYMIZER_OUTPUT_TOKENS, output_reserve)`` — without it, a
       small fact set's computed reserve can sit below the structural
       floor, so the packer would accept facts up to a prompt size that
       leaves less than ``_MIN_ANONYMIZER_OUTPUT_TOKENS`` of the envelope,
       tripping ``anonymize_transcript``'s own clamp WARNING on a slice
       the packer believed was correctly sized.
    4. A single fact that alone exceeds the envelope is still emitted as
       its own slice (facts are atomic) — the clamp inside
       :func:`anonymize_transcript` then applies to that slice's call.
    5. Empty ``facts`` -> ``[[]]`` (one empty slice — the call still
       runs; there may be nothing to anonymize but the caller still
       needs a result).
    """
    if not facts:
        return [[]]

    base_prompt = _render_anonymize_prompt(
        [],
        tokenizer,
        scrub=scrub,
        transcript="",
        user_prompt_template=user_prompt_template,
        system_prompt=system_prompt,
    )
    base = estimate_tokens(base_prompt, tokenizer)

    def _fact_surfaces(fact: dict) -> set[str]:
        return {str(fact.get(field, "")) for field in ("subject", "object") if fact.get(field)}

    slices: list[list[dict]] = []
    current: list[dict] = []
    current_fragment_tokens = 0
    current_surfaces: set[str] = set()
    current_reserve = 0

    for fact in facts:
        fragment_tokens = estimate_tokens(json.dumps(fact), tokenizer) + _FACT_SEPARATOR_TOKENS
        new_surfaces = _fact_surfaces(fact) - current_surfaces
        new_reserve = sum(_mapping_reserve_tokens(s, tokenizer) for s in new_surfaces)

        projected_reserve = max(
            _MIN_ANONYMIZER_OUTPUT_TOKENS,
            _OUTPUT_JSON_ENVELOPE_TOKENS + current_reserve + new_reserve,
        )
        projected = base + current_fragment_tokens + fragment_tokens + projected_reserve
        if current and projected > token_envelope:
            slices.append(current)
            current = [fact]
            current_fragment_tokens = fragment_tokens
            current_surfaces = _fact_surfaces(fact)
            current_reserve = sum(_mapping_reserve_tokens(s, tokenizer) for s in current_surfaces)
        else:
            current.append(fact)
            current_fragment_tokens += fragment_tokens
            current_surfaces |= new_surfaces
            current_reserve += new_reserve

    if current:
        slices.append(current)
    return slices


def anonymize_transcript(
    facts: list[dict],
    model,
    tokenizer,
    *,
    scrub: set[str] | frozenset[str],
    transcript: str = "",
    token_envelope: int = _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
    configured_envelope: int | None = None,
    free_mib: float | None = None,
    temperature: float = _DEFAULT_ANONYMIZER_TEMPERATURE,
    seed: int | None = None,
    user_prompt_template: str,
    system_prompt: str,
    speaker_id: str | None = None,
    speaker_anchor_template: str = "",
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
    "anonymized_transcript": [...]}`` envelope) OR on the inconsistent
    shape: the model named something in ``mapping`` over a non-empty
    input ``transcript`` but returned a missing/empty
    ``anonymized_transcript``.  Callers MUST never fall back to the
    original real-name transcript on this signal.

    An empty/missing ``anonymized_transcript`` is otherwise LEGITIMATE —
    there was nothing to rewrite — in either of two cases: the input
    ``transcript`` is empty (the graph tier: no transcript exists at
    all), or the model's ``mapping`` is empty (``scrub`` defines what
    needs anonymizing; "nothing matched" is a valid outcome).  In both
    legitimate-empty cases this function returns ``(mapping, "",
    raw_output)`` — ``mapping`` may itself be empty or (in the
    empty-transcript case only) non-empty; the caller (:func:`anonymize`)
    proceeds with the ORIGINAL argument transcript rather than a model
    artifact, exactly as :func:`opted_out_contract` already does.

    ``anonymized_transcript`` in the model's response is a JSON array of
    turn strings (one element per turn) per the ``configs/prompts/
    anonymization.txt`` contract — this keeps every turn on its own line
    so a multi-turn rewrite can never contain a literal newline inside a
    JSON string value (illegal per RFC 8259, the root cause of an
    observed multi-turn parse failure).  The array is joined with
    ``"\\n"`` back into a single transcript string before being returned.
    A plain ``str`` value is also still accepted and returned unchanged,
    since the model may emit the pre-contract shape.  A NON-EMPTY array
    containing a non-``str`` element is always fail-closed (the shape
    check) regardless of the transcript/mapping state above; an EMPTY
    array is treated as "missing" and falls into the legitimate-empty
    rule.

    The raw model output is the third element so the calibration phase
    trace can record it without re-running the call.  ``anonymized_transcript``
    is ``""`` whenever ``mapping`` is ``None`` (the two fail-closed
    together).

    ``scrub`` is required — no implicit default.  An omitting caller would
    silently anonymize against a hidden default on a security-critical
    egress path; the single declared default is
    ``SanitizationConfig.scrub`` (``paramem/server/config.py``), which
    every caller of this function threads down explicitly.

    ``token_envelope`` is the total (prompt + output) token budget one
    local call may occupy — the EFFECTIVE envelope when called from
    :func:`anonymize` (already clamped to live free VRAM there; see
    :func:`~paramem.utils.vram_guard.effective_token_envelope`), or the
    configured/default value verbatim for any other caller (calibration,
    tests).  ``max_new_tokens`` is DERIVED —
    ``max(_MIN_ANONYMIZER_OUTPUT_TOKENS, token_envelope -
    actual_prompt_tokens)`` — never a second, independently-configured
    knob.  When the prompt alone leaves less than the estimated output
    reserve (:func:`_output_reserve_tokens`) or the structural floor
    (:data:`_MIN_ANONYMIZER_OUTPUT_TOKENS`), the call proceeds anyway
    with the clamped allowance and logs a WARNING naming the overshoot —
    a truncated response is not a leak, it fails to parse and the chain
    fails closed; the warning is the operator's signal that a payload was
    oversized for the effective envelope.

    ``configured_envelope`` is the operator-configured CEILING this call's
    ``token_envelope`` was clamped from — ``None`` when the caller applied
    no VRAM clamp (``token_envelope`` IS the ceiling in that case).  Named
    separately in the overshoot WARNING so an operator can tell "the
    configured ceiling was itself too small" apart from "live VRAM clamped
    a normally-sufficient ceiling down" — the two have different remedies
    (raise the config value vs. free VRAM / reduce concurrent GPU load).

    ``free_mib`` is the live free-VRAM reading
    :func:`~paramem.utils.vram_guard.effective_token_envelope` took when
    it computed ``token_envelope`` — ``None`` when no measurement was
    taken (CPU, or the driver query failed).  Logged only, for the
    telemetry line below; never used in the ``max_new_tokens`` arithmetic
    (the clamp already happened at the caller).

    ``seed`` is forwarded verbatim to :func:`~paramem.evaluation.recall.
    generate_answer`.  At the default ``temperature=0.0`` (greedy
    decoding) it is a strict no-op.

    ``user_prompt_template``/``system_prompt`` are the ALREADY-LOADED
    prompt contents — the caller resolves
    ``configs/prompts/anonymization.txt`` (session tier) or
    ``anonymization_facts.txt`` (graph tier) /
    ``anonymization_system.txt`` (with calibration overrides and
    provenance recording) via
    :func:`~paramem.graph.prompts._load_prompt` and passes the text in;
    this function never touches the filesystem.

    THE only production caller of this function is :func:`anonymize`'s
    per-slice loop — every path that anonymizes for cloud egress reaches
    the model through the one chain, never this function directly.

    ``speaker_id``/``speaker_anchor_template`` are forwarded verbatim to
    :func:`_render_anonymize_prompt` — see that function's docstring for
    the ``{speaker_anchor_section}`` slot contract and the
    ``speaker_id``-gated rendering rule.
    """
    formatted = _render_anonymize_prompt(
        facts,
        tokenizer,
        scrub=scrub,
        transcript=transcript,
        user_prompt_template=user_prompt_template,
        speaker_id=speaker_id,
        speaker_anchor_template=speaker_anchor_template,
        system_prompt=system_prompt,
    )

    # Payload-size telemetry at the boundary into the guarded generate call
    # below — mirrors the extraction-side plaus_filter instrumentation
    # (paramem.graph.extractor.judge_plausibility, "plaus_filter prompt:
    # chars=%d tokens=%d max_new_tokens=%d"). This is the ONLY
    # tokenization of `formatted` done purely for the count (generate_answer
    # tokenizes again internally to build the generation tensors — that
    # second call is unavoidable, it is not a re-tokenization for the same
    # purpose). Discriminates the fold-tier caller (large chunk payload)
    # from the session-tier caller (single-turn payload) at the one call
    # site both funnel through.
    prompt_tokens = estimate_tokens(formatted, tokenizer)
    transcript_tokens = estimate_tokens(transcript, tokenizer) if transcript else 0
    output_reserve = _output_reserve_tokens(facts, tokenizer, transcript_tokens=transcript_tokens)
    desired_reserve = max(_MIN_ANONYMIZER_OUTPUT_TOKENS, output_reserve)
    raw_allowance = token_envelope - prompt_tokens
    max_new_tokens = max(_MIN_ANONYMIZER_OUTPUT_TOKENS, raw_allowance)
    if raw_allowance < desired_reserve:
        # `configured_envelope` names the operator ceiling separately from
        # `token_envelope` (the effective, possibly VRAM-clamped value) so
        # an operator can tell "the ceiling itself was too small" apart
        # from "live VRAM clamped a normally-sufficient ceiling down" —
        # they read identically here whenever no clamp applied (the
        # `or token_envelope` fallback for direct callers that never went
        # through `anonymize`'s clamp).
        ceiling = configured_envelope if configured_envelope is not None else token_envelope
        logger.warning(
            "anonymize_transcript: prompt (%d tok) leaves %d tok inside the %d-token "
            "effective envelope (operator ceiling %d tok), less than the estimated "
            "output reserve (%d tok) — proceeding with the clamped allowance (%d tok); "
            "response may truncate and fail closed.",
            prompt_tokens,
            raw_allowance,
            token_envelope,
            ceiling,
            desired_reserve,
            max_new_tokens,
        )

    # `free_mib` is -1 when no live-VRAM measurement was taken (CPU, or
    # the driver query failed inside effective_token_envelope) — a
    # documented sentinel for THIS log field only (never fed into a
    # budgeting decision, unlike the retired -1 token-count sentinel this
    # project otherwise avoids).
    free_mib_log = int(free_mib) if free_mib is not None else -1
    logger.info(
        "anonymize_transcript prompt: chars=%d tokens=%d max_new_tokens=%d "
        "effective_envelope=%d free_mib=%d",
        len(formatted),
        prompt_tokens,
        max_new_tokens,
        token_envelope,
        free_mib_log,
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
    logger.debug("Anonymization raw: %s", raw[:500])

    try:
        json_str = _extract_json_block(raw)
        data = json.loads(json_str)
        # Envelope + mapping-type checks come BEFORE the rewrite check:
        # validity of an empty/missing rewrite depends on whether
        # `mapping` is itself empty (the legitimate-empty rule below), so
        # `mapping` must be read and type-checked first.
        if not isinstance(data, dict) or "mapping" not in data:
            logger.warning("Anonymization returned unexpected format")
            return None, "", raw
        raw_mapping = data["mapping"]
        if not isinstance(raw_mapping, dict):
            logger.warning("Anonymization 'mapping' is not a dict")
            return None, "", raw

        anon_transcript = data.get("anonymized_transcript")
        if isinstance(anon_transcript, list):
            # Contract shape: one turn string per element (see
            # configs/prompts/anonymization.txt) — join back into a
            # single transcript.  A NON-EMPTY array with a non-str
            # element is always fail-closed (malformed shape); an EMPTY
            # array falls through to the legitimate-empty rule below,
            # exactly like a missing key.
            if anon_transcript and not all(isinstance(t, str) for t in anon_transcript):
                logger.warning("Anonymization returned a malformed anonymized_transcript array")
                return None, "", raw
            anon_transcript = "\n".join(anon_transcript) if anon_transcript else ""
        elif anon_transcript is None:
            anon_transcript = ""
        elif not isinstance(anon_transcript, str):
            logger.warning("Anonymization 'anonymized_transcript' is not a str, list, or null")
            return None, "", raw

        if not anon_transcript.strip():
            # An empty/missing rewrite is legitimate whenever there was
            # nothing to rewrite: the input transcript is empty (graph
            # tier — no transcript exists), or the model's own mapping
            # came back empty ("nothing in scope" is already the
            # accepted verdict on the facts surface).  It narrows to
            # fail-closed ONLY on the inconsistent shape: the model
            # named something (non-empty raw_mapping) over a non-empty
            # input transcript but authored no rewrite.
            if transcript and raw_mapping:
                logger.warning(
                    "Anonymization named entities but returned missing/empty "
                    "anonymized_transcript over a non-empty transcript"
                )
                return None, "", raw
            return raw_mapping, "", raw

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
      ``reverse`` are ``{}``; ``facts`` is the input ``facts`` verbatim
      (identity substitution downstream via
      :func:`~paramem.cloud.placeholders.insert_placeholders` over this
      empty ``forward``).
    * ``"failed"`` — fail-closed: every slice's local anonymizer call
      failed to parse or returned the inconsistent-shape rewrite (step
      3), OR every slice's domain-scoped privacy guard fired (step 6).
      ``forward`` / ``reverse`` / ``facts`` are empty; ``anon_transcript``
      is ``""``.  Callers must NEVER fall back to the original real-name
      transcript on this status, and must NOT derive an anonymized fact
      array from this contract.  ``failure`` distinguishes which of the
      two causes dominated — see below.
    * ``"ok"`` — at least one slice's anonymizer ran (and, when
      ``identity_domain`` was supplied, reconciliation and the privacy
      guard both passed for that slice).  ``forward`` / ``reverse`` may
      still be empty — a legitimate verdict ("ran, found nothing in
      scope"), not a failure; egress proceeds.  ``facts`` is the input
      ``facts`` MINUS the facts of every fail-closed slice.

    ``failure`` — ``None`` except when ``status == "failed"``, where it is
    always one of:

    * ``"parse"`` — every slice's local anonymizer call itself failed: a
      JSON parse failure, or the inconsistent-shape rewrite (step 3).  No
      mapping was produced at all.
    * ``"guard"`` — the domain-scoped fail-closed guard fired on at least
      one slice and no slice produced a usable table (step 6): the
      anonymizer named something, but nothing survived to the final
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

    ``facts`` is the (real-name, un-substituted) fact subset cleared for
    egress — every production reader derives the anonymized array on
    demand via :func:`~paramem.cloud.placeholders.insert_placeholders`
    over ``facts`` and ``forward``, never storing the substituted array
    here (this function does not build it — see :func:`anonymize`'s
    docstring).

    ``slices`` is the number of local ``anonymize_transcript`` calls this
    contract cost (1 for every non-graph-tier call — a transcript is
    atomic and never sliced); ``slices_failed`` is how many of those
    calls were dropped fail-closed (parse or guard) and therefore
    contributed none of their facts to ``facts``.
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
    facts: list[dict] = field(default_factory=list)
    slices: int = 1
    slices_failed: int = 0


def opted_out_contract(transcript: str, *, facts: list[dict]) -> AnonymizedContract:
    """The ``status="opted_out"`` shape — the operator-opt-out (``scrub``
    empty) result, with ``transcript`` egressing verbatim and ``facts``
    egressing verbatim (identity substitution via
    :func:`~paramem.cloud.placeholders.insert_placeholders` over the
    empty ``forward`` map).

    THE single constructor for this shape.  :func:`anonymize`'s own
    ``scrub``-empty branch uses it, threading its own ``facts`` argument
    through unchanged.  :func:`~paramem.graph.flows.anonymize_turn`'s
    early ``if not scrub:`` return — which short-circuits BEFORE the
    local extraction pass that would otherwise anchor the anonymizer, so
    no compute is wasted anchoring a call that will never run — also uses
    this constructor, but passes ``facts=[]`` explicitly and correctly:
    that call site's docstring already states a caller deriving the
    anonymized fact array from its contract "would always get ``[]``"
    (chat egress never reads ``payload.facts`` — its only output is the
    anonymized transcript).  Before this helper existed the two sites
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
        facts=list(facts),
        slices=0,
        slices_failed=0,
    )


def _index_identity_domain(
    identity_domain: Iterable[str] | None,
) -> tuple[dict[str, str], set[str]]:
    """Build the canonical-form -> domain-surface index for identity
    reconciliation ONCE per :func:`anonymize` call, above its per-slice
    loop.

    ``identity_domain`` is constant across a call's slices (the graph
    tier's ``chunk_nodes``, up to ``max_entities_per_pass`` — typically
    50 entries) — re-canonicalising it per slice would apply the same
    transformation N times to one value.  Returns ``(canon_to_domain,
    ambiguous_canon)``: a domain entry whose canonical form collides with
    an earlier one is a genuine ambiguity, recorded in ``ambiguous_canon``
    rather than silently picking one.

    ``identity_domain is None`` (no domain to reconcile against — the
    session tier / chat egress / calibration) returns two empty
    collections; the caller gates :func:`_reconcile_to_domain` and
    :func:`_domain_guard_fires` on ``identity_domain is not None``, so
    this function is never even called with a domain in that case except
    to produce the empty pair.
    """
    canon_to_domain: dict[str, str] = {}
    ambiguous_canon: set[str] = set()
    if identity_domain is None:
        return canon_to_domain, ambiguous_canon
    for d in identity_domain:
        c = canonical(str(d))
        if c in canon_to_domain and canon_to_domain[c] != d:
            # Two distinct domain entries canonicalizing identically —
            # fail closed on this entry rather than silently pick one.
            ambiguous_canon.add(c)
        else:
            canon_to_domain[c] = d
    return canon_to_domain, ambiguous_canon


def _reconcile_to_domain(
    mapping: dict[str, str],
    canon_to_domain: dict[str, str],
    ambiguous_canon: set[str],
) -> tuple[dict[str, str], int]:
    """Re-key ``mapping`` (already normalized, ``{real: placeholder}``)
    onto ``canon_to_domain``'s domain surfaces, preserving the model's
    own placeholder verbatim.

    Every key is folded through :func:`~paramem.utils.identity.canonical`
    and matched against ``canon_to_domain``.  A key whose canonical form
    is ambiguous, or has no domain match, is dropped and counted.
    Returns ``(reconciled_mapping, dropped_count)``.

    Extracted, unchanged in behaviour, from :func:`anonymize`'s
    pre-slicing body — see :func:`_index_identity_domain` for why the
    domain index itself is built once, outside this function (and once
    per :func:`anonymize` call, not once per slice this function is
    invoked for).
    """
    reconciled: dict[str, str] = {}
    dropped = 0
    for real, placeholder in mapping.items():
        c = canonical(real)
        if c in ambiguous_canon or c not in canon_to_domain:
            dropped += 1
            continue
        reconciled[canon_to_domain[c]] = placeholder
    return reconciled, dropped


def _domain_guard_fires(llm_mapping: dict, mapping: dict, facts: list[dict]) -> bool:
    """The domain-scoped fail-closed guard (step 6 of :func:`anonymize`'s
    chain, evaluated per slice): the anonymizer named something
    (``llm_mapping`` non-empty) but the reconciled ``mapping`` came back
    empty AND ``facts``' subject/object endpoints contain a non-speaker
    name.

    The guard domain is deliberately derived from ``facts`` — the SLICE's
    own facts, never ``identity_domain`` — because the two differ:
    ``identity_domain`` (the rekey domain) is a caller-supplied node list
    that may be trimmed to a size cap; the guard domain is the facts'
    actual subject/object endpoints, a subset of that list once trimming
    drops nodes whose only surviving edges fell outside the chunk/slice.
    Fusing the two would silently reject a slice whose surviving edges
    are all speaker-only just because the (larger, untrimmed) node list
    still lists a non-speaker node — a false-positive regression, not a
    privacy fix.
    """
    if not llm_mapping or mapping:
        return False
    endpoint_names = {str(f.get(field, "")) for f in facts for field in ("subject", "object")}
    endpoint_names = {n for n in endpoint_names if n and not is_speaker_id(n)}
    return bool(endpoint_names)


def anonymize(
    facts: list[dict],
    model,
    tokenizer,
    *,
    transcript: str,
    scrub: set[str] | frozenset[str],
    speaker_name: str | None = None,
    speaker_id: str | None = None,
    speaker_anchor_template: str = "",
    identity_domain: Iterable[str] | None = None,
    token_envelope: int = _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
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

    ``speaker_id`` — THIS session's own ``speaker{{N}}`` token — fills the
    anonymizer prompt's ``{speaker_anchor_section}`` slot (fold-onto-token
    anonymization) together with ``speaker_anchor_template`` (the
    caller-loaded ``configs/prompts/anonymization_speaker_anchor.txt``
    contents): it tells the anonymizer LLM which placeholder a coreferring
    real-name mention of the user may fold onto, so the mapping it returns
    can bind directly onto the session's own token instead of minting a
    separate ``Person_N`` for the user. Threaded verbatim into every
    slice's :func:`anonymize_transcript` call; ``speaker_id=None`` (the
    graph tier, which has no single-session speaker, and any calibration
    caller that omits it) renders the slot as ``""`` regardless of
    ``speaker_anchor_template`` — see :func:`_render_anonymize_prompt`'s
    docstring. Every caller of THIS function is responsible for its own
    anchor-validity gate before passing ``speaker_id`` in: the chat-egress
    caller (:func:`~paramem.graph.flows.anonymize_turn`) forwards it only
    when the value satisfies :func:`~paramem.utils.identity.is_speaker_id`
    (a well-shaped id is sufficient — an anonymous-enrolled speaker with
    no resolvable display name still KEEPS the anchor, owner ruling: their
    session facts and cloud payloads stay in token space exactly like a
    named speaker's) — this function does not re-validate that invariant,
    it only renders whatever ``speaker_id`` it is given. Orthogonal to
    ``speaker_name`` (step 8 below): that seeds the DISPLAY name's own
    forward-map coverage; this tells the model which TOKEN the display name
    (if mentioned) should fold onto.

    **Slicing decision (no flag, derived from the arguments): slice iff
    ``transcript`` is empty.**  A transcript is atomic — repeating it in
    every slice would multiply its cost and make the model-authored
    rewrite unmergeable (which slice's rewrite would win?).  So the graph
    tier (``transcript=""``) packs ``facts`` into fact-boundary slices via
    :func:`_slice_facts_to_envelope` and makes one local call per slice;
    the session tier and chat egress (``transcript`` non-empty) make
    exactly one call over the whole ``facts`` list — chunk topology and
    cloud-call count are unchanged either way, this only governs LOCAL
    ``generate()`` calls.

    **Dynamic VRAM clamp (owner-approved 2026-07-28):** ``token_envelope``
    is the operator-configured CEILING, not a guarantee that live free
    VRAM can support it at call time — free VRAM varies within one fold
    (live evidence: a packer-correct 8,192-token call still faulted
    "device not ready" with only 1,191 MiB free, minutes after a
    4,249-token call passed with 1,407 MiB free).  This function measures
    free VRAM exactly ONCE, at entry, via
    :func:`~paramem.utils.vram_guard.effective_token_envelope`, and
    threads the resulting effective envelope — ``min(token_envelope,
    tokens supportable by that one measurement)`` — to BOTH
    :func:`_slice_facts_to_envelope` and every slice's
    :func:`anonymize_transcript` call below.  There is deliberately no
    mid-call re-measurement: each ``generate()`` call frees its own KV
    cache on exit (:func:`~paramem.utils.vram_guard.safe_empty_cache`
    inside :func:`~paramem.utils.vram_guard.vram_scope`), so free VRAM
    typically recovers between slices — an entry-time measurement is
    conservative (never optimistic) for every later slice, not just the
    first.  No CUDA available -> the effective envelope equals the
    configured one (a strict passthrough; see
    :func:`~paramem.utils.vram_guard.effective_token_envelope`'s
    docstring) — the CPU test suite never needs a GPU to exercise this
    function.

    Fixed sequence:

    1. ``scrub`` empty -> ``status="opted_out"``, empty tables,
       ``anon_transcript=transcript`` verbatim, ``facts`` verbatim.  A
       caller deriving the anonymized fact array gets the identity
       (empty-mapping) substitution.
    2. Per slice, :func:`anonymize_transcript` (``speaker_id=speaker_id``,
       forwarded verbatim to that call's own :func:`_render_anonymize_prompt`
       — see its docstring for the ``{speaker_id}`` slot contract) ->
       ``(llm_mapping, model_anon_transcript, raw)``.
    3. ``llm_mapping is None`` -> that slice's facts are dropped
       (``slices_failed`` += 1, ``failure`` candidate ``"parse"``) and the
       loop continues to the next slice.  Never falls back to the
       real-name transcript for that slice.
    4. :func:`~paramem.cloud.placeholders._normalize_anonymization_mapping`
       — per slice; ``norm_stats`` accumulates across every slice and is
       a LIVE signal callers persist (``{"inverted": N, "dropped": N}``).
    5. **Identity reconciliation** (only when ``identity_domain is not
       None`` — the graph tier's pre-step, generalized as data), per
       slice, via :func:`_reconcile_to_domain` against the domain index
       :func:`_index_identity_domain` built ONCE above the loop (the
       domain itself is constant across slices).  A miss or an ambiguous
       multi-match is dropped and counted into ``rekey_dropped``
       (accumulated across slices).  ``identity_domain=None`` (session
       tier / chat egress / calibration) skips this step entirely —
       those mapping keys are free-text surfaces with no closed domain to
       reconcile against.
    6. **Domain-scoped fail-closed guard** (:func:`_domain_guard_fires`),
       per slice — fires ONLY when ``identity_domain is not None`` AND
       that slice's reconciled-from mapping was non-empty AND the
       reconciled mapping came back empty AND that slice's ``facts``'
       subject/object endpoints contain a non-speaker name.  On fire,
       that slice's facts are dropped (``slices_failed`` += 1, ``failure``
       candidate ``"guard"``) and the loop continues.  See
       :func:`_domain_guard_fires`'s docstring for why the guard domain
       is derived from the slice's own facts, never ``identity_domain``.
    7. Surviving slices' per-slice mappings are merged into one table.
       Cross-KEY dedup — first-wins on the CANONICAL real-value key — is
       scoped to entries carried over from an EARLIER slice only (an
       entity seen in a *prior* slice keeps that slice's placeholder
       decision): the test is against a ``canon_seen`` snapshot taken
       before each slice starts, never against an entry the SAME slice is
       in the middle of adding.  Within one slice, two literal real-value
       keys that canonicalize identically (e.g. ``"José García"`` /
       ``"Jose Garcia"``, a diacritic-fold collision) both survive, each
       keeping its own model-assigned placeholder — matching pre-slicing
       single-call behavior, where no cross-key canon dedup existed at
       all.  (A slice-boundary-blind, unconditional canon dedup here
       would silently drop the second spelling from the merged table;
       since :func:`~paramem.cloud.placeholders._substitute_whole_words`
       matches EXACTLY, the dropped literal would then egress UNMASKED —
       a code-review-caught regression this scoping fixes.)
       Placeholder-VALUE collisions are handled differently and are NOT
       scoped to prior slices: ``used_placeholders`` is checked and
       updated LIVE, so two DIFFERENT real values sharing the SAME
       placeholder — whether from the same slice or different slices —
       are both re-minted onto fresh, unique placeholders via
       :func:`~paramem.cloud.placeholders.mint_placeholder` (the
       uniqueness contract the prompt teaches, now enforced within AND
       across slices).  This means the within-call duplicate-placeholder
       scenario IS closed here: a slice whose model emits ``{"Alice":
       "Person_1", "Pat": "Person_1"}`` gets ``"Pat"`` re-minted onto
       ``"Person_2"`` before either entry ever reaches
       :func:`~paramem.cloud.placeholders._build_anonymization_mapping`,
       so :func:`~paramem.cloud.placeholders.invert_forward_mapping`'s own
       first-wins tie-break is never actually exercised by this call
       path.
    8. :func:`~paramem.cloud.placeholders._build_anonymization_mapping`
       (``speaker_name=speaker_name``), run ONCE over the merged table ->
       ``(forward, reverse)``.  THE speaker-value guard now applies on
       every path by construction — no code path is left that inverts an
       unfiltered forward map.
    9. :func:`~paramem.cloud.placeholders._declared_placeholder_tokens`
       (``reverse``) -> ``declared`` — the CORE declared vocabulary (no
       Cloud bindings yet at this stage; see
       :meth:`~paramem.cloud.deanonymize.CloudScope.response` for the
       union used at deanon time).

    Terminal status: every slice fail-closed (``slices_failed ==
    slices``) -> ``status="failed"``, ``failure="guard"`` if any slice's
    guard fired else ``"parse"``, ``facts=[]``.  Otherwise
    ``status="ok"``, ``facts`` = the input ``facts`` minus every
    fail-closed slice's facts.

    This function does NOT build the anonymized fact array itself — every
    production reader derives it on demand instead, via
    :func:`~paramem.cloud.placeholders.insert_placeholders` over this
    contract's ``facts`` and ``forward`` map: the ``enrich`` stage
    (:mod:`paramem.graph.stage_enrich`) and ``request_graph_enrichment``
    (:mod:`paramem.graph.extractor`).

    ``user_prompt_template``/``system_prompt`` are the caller-loaded
    ``anonymization.txt``/``anonymization_facts.txt`` /
    ``anonymization_system.txt`` contents — see
    :func:`anonymize_transcript`'s docstring for why this package never
    loads a prompt file itself.
    """
    if not scrub:
        return opted_out_contract(transcript, facts=facts)

    # ONE measurement for the whole call — see the docstring's "Dynamic
    # VRAM clamp" paragraph. `effective_envelope` is threaded to both the
    # packer and every slice's anonymize_transcript call below;
    # `token_envelope` itself is kept unchanged so the WARNING inside
    # anonymize_transcript can still name the operator's configured
    # ceiling alongside the clamped value.
    effective_envelope, free_mib = effective_token_envelope(token_envelope)

    if transcript:
        # Atomic — a transcript is never split across slices.
        slices = [facts]
    else:
        slices = _slice_facts_to_envelope(
            facts,
            tokenizer,
            scrub=scrub,
            user_prompt_template=user_prompt_template,
            system_prompt=system_prompt,
            token_envelope=effective_envelope,
        )

    canon_to_domain, ambiguous_canon = _index_identity_domain(identity_domain)

    accepted_facts: list[dict] = []
    merged: dict[str, str] = {}
    canon_seen: dict[str, str] = {}
    used_placeholders: set[str] = set()
    parse_failures = 0
    guard_failures = 0
    norm_inverted = 0
    norm_dropped = 0
    rekey_dropped_total = 0
    raw_parts: list[str] = []
    model_anon_transcript = ""

    for slice_facts in slices:
        llm_mapping, slice_anon_transcript, raw = anonymize_transcript(
            slice_facts,
            model,
            tokenizer,
            transcript=transcript,
            scrub=scrub,
            token_envelope=effective_envelope,
            configured_envelope=token_envelope,
            free_mib=free_mib,
            seed=seed,
            user_prompt_template=user_prompt_template,
            system_prompt=system_prompt,
            speaker_id=speaker_id,
            speaker_anchor_template=speaker_anchor_template,
        )
        if raw:
            raw_parts.append(raw)
        if llm_mapping is None:
            parse_failures += 1
            continue

        mapping, norm_stats = _normalize_anonymization_mapping(llm_mapping)
        norm_inverted += norm_stats["inverted"]
        norm_dropped += norm_stats["dropped"]

        if identity_domain is not None:
            mapping, dropped = _reconcile_to_domain(mapping, canon_to_domain, ambiguous_canon)
            rekey_dropped_total += dropped
            if _domain_guard_fires(llm_mapping, mapping, slice_facts):
                guard_failures += 1
                continue

        # Snapshot BEFORE processing this slice's mapping: the cross-KEY
        # canon dedup below is scoped to entries carried over from an
        # EARLIER slice only, tested against this snapshot — never
        # against an entry this SAME slice is in the middle of adding.
        # An unconditional (live) canon_seen check would drop the SECOND
        # of two literal real-value keys that canonicalize identically
        # but both arrive within one call (e.g. "José García" / "Jose
        # Garcia" — diacritic-fold collision) — since
        # _substitute_whole_words matches EXACTLY, the dropped literal
        # would then egress UNMASKED (the B1 regression a code review
        # caught). Scoping to the snapshot restores the pre-slicing
        # single-call behavior, where no cross-key canon dedup existed
        # at all: within one slice, both spellings survive with their
        # own placeholder.
        canon_seen_before_slice = dict(canon_seen)
        for real, placeholder in mapping.items():
            c = canonical(real)
            if c in canon_seen_before_slice:
                # First-wins: an entity seen in an EARLIER slice keeps
                # that slice's placeholder decision.
                continue
            if placeholder in used_placeholders and not is_speaker_id(placeholder):
                # Placeholder-VALUE collisions are NOT scoped to prior
                # slices — used_placeholders is checked and updated LIVE,
                # catching a collision against a prior slice OR an entry
                # already processed earlier in THIS SAME slice (the
                # within-call duplicate-placeholder closure: a slice
                # whose model emits two DIFFERENT real values on the SAME
                # placeholder, e.g. {"Alice": "Person_1", "Pat":
                # "Person_1"}, gets the second re-minted here before
                # either ever reaches _build_anonymization_mapping).
                # Re-mint onto the same prefix so the merged table's
                # placeholders stay unique either way.
                #
                # Carve-out: a speaker-id-shaped placeholder is EXEMPT from
                # this uniqueness enforcement. Fold-onto-token anonymization
                # (configs/prompts/anonymization_speaker_anchor.txt)
                # deliberately authorizes MULTIPLE distinct real-value keys
                # (e.g. "Priya" and "Priya Sharma", both self-naming
                # variants the speaker used in the same session) to map
                # onto the SAME speaker{N} anchor — that is the fold, not a
                # collision, and re-minting the second one onto a fresh
                # Thing_N would leave it unresolved to the anchor and
                # scrubbed as an unrelated entity instead.
                placeholder = mint_placeholder(
                    used_placeholders, placeholder_prefix(placeholder) or "Thing"
                )
            canon_seen[c] = real
            used_placeholders.add(placeholder)
            merged[real] = placeholder

        accepted_facts.extend(slice_facts)
        if transcript:
            model_anon_transcript = slice_anon_transcript

    slices_failed = parse_failures + guard_failures
    # `slices` is never empty: `_slice_facts_to_envelope` always returns
    # at least one (possibly empty-facts) slice, and the atomic
    # (transcript-bearing) branch above sets `slices = [facts]`
    # unconditionally — so the terminal check needs no truthiness guard.
    if slices_failed == len(slices):
        return AnonymizedContract(
            status="failed",
            forward={},
            reverse={},
            anon_transcript="",
            declared=frozenset(),
            norm_stats={"inverted": norm_inverted, "dropped": norm_dropped},
            rekey_dropped=rekey_dropped_total,
            raw="\n".join(raw_parts),
            failure="guard" if guard_failures else "parse",
            facts=[],
            slices=len(slices),
            slices_failed=slices_failed,
        )

    forward, reverse = _build_anonymization_mapping(merged, speaker_name=speaker_name)
    declared = frozenset(_declared_placeholder_tokens(reverse))

    return AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        # Argument-sourced fallback, never a model artifact — matches
        # opted_out_contract's discipline. On the sliced (graph-tier)
        # path both terms are "" by construction (transcript is "").
        anon_transcript=model_anon_transcript or transcript,
        declared=declared,
        norm_stats={"inverted": norm_inverted, "dropped": norm_dropped},
        rekey_dropped=rekey_dropped_total,
        raw="\n".join(raw_parts),
        facts=accepted_facts,
        slices=len(slices),
        slices_failed=slices_failed,
    )
