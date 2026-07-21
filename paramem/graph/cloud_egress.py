"""The one anonymize -> cloud -> deanonymize round-trip contract.

``paramem.graph.placeholders`` is the primitive kit: mint, substitute,
normalize, invert, resolution map, totality, apply-bindings — model-free,
IO-free (see that module's docstring: "Nothing here has an opinion about
WHERE the table comes from ... or WHAT happens to the result"). Every path
that hands text/facts to a cloud model and restores real identities
afterward composes those primitives into the SAME two-step chain:
anonymize (:func:`anonymize_for_cloud`) then deanonymize
(:func:`deanonymize_facts` / :func:`deanonymize_response_text`). This
module IS that chain, once — every caller (session-tier extraction
(``_sota_pipeline``), graph-tier enrichment (``_graph_enrich_with_sota``),
chat egress (``extract_and_anonymize_for_cloud``), and their calibration
harnesses) reaches a reverse map, a totality gate, and an ``observed``
scope ONLY through the functions here.

Boundary: ``placeholders.py`` = the primitive kit (no model, no network);
``cloud_egress.py`` (here) = the one round-trip contract, including the
local-model anonymizer call (:func:`anonymize_with_local_model`) and the
generic model-output JSON-envelope parser (:func:`_extract_json_block`) it
depends on; ``paramem.graph.extractor`` / ``paramem.training.consolidation``
/ ``paramem.server.inference`` = orchestrators owning only their own
prompt render and their own reaction to a rejection.

``_extract_json_block`` and the ``_DEFAULT_FILTER_*`` defaults live here
rather than in ``extractor.py`` (where every OTHER SOTA-response parser
still lives) because ``extractor.py`` imports FROM this module
(:func:`anonymize_for_cloud`, :class:`CloudScope`, :func:`deanonymize_facts`,
:func:`deanonymize_response_text`) — the dependency is one-directional.
Placing either in ``extractor.py`` and importing back here would be a
genuine import cycle: ``extractor.py`` would need this module fully
initialized to resolve its own top-level import, and this module would
need ``extractor.py`` fully initialized to resolve ITS top-level import,
with neither able to finish first. A function-local import to dodge that
cycle is rejected — it would leave half the contract split across two
files for no reason other than import ordering. ``extractor.py`` imports
both back for its own remaining SOTA-response parsers
(``_parse_facts_response``, ``_parse_drop_set``, ``_parse_enrichment_delta``,
``_filter_with_sota``, ``_graph_enrich_with_sota``, ...).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paramem.evaluation.recall import generate_answer
from paramem.graph.placeholders import (
    _apply_bindings,
    _build_anon_facts,
    _build_anonymization_mapping,
    _check_mapping_totality,
    _contains_declared_token,
    _declared_placeholder_tokens,
    _normalize_anonymization_mapping,
    _resolution_map,
    deanonymize_text,
)
from paramem.graph.prompts import _load_prompt
from paramem.graph.schema import SessionGraph
from paramem.models.loader import adapt_messages
from paramem.server.vram_guard import vram_scope
from paramem.utils.identity import canonical, is_speaker_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared SOTA-round-trip defaults. Defined here (not in extractor.py) so
# extractor.py's many SOTA-call sites can import a single source of truth
# without creating the cycle described in the module docstring — this
# module is imported BY extractor.py, never the reverse.
# ---------------------------------------------------------------------------

# Governs both the anonymizer's local generate() call (via
# ``anonymize_for_cloud``'s default) and every other SOTA round trip
# (enrichment, plausibility) that extractor.py still owns directly.
# 8192: the enrichment/plausibility prompt's output volume scales with
# input fact count; 2048 truncated dense chunks and broke the parse.
_DEFAULT_FILTER_MAX_TOKENS = 8192
# Deterministic by default; threaded to every provider call so Anthropic
# and OpenAI-compatible filters match exactly.
_DEFAULT_FILTER_TEMPERATURE = 0.0
# Per-call timeout for SOTA enrichment / plausibility / anonymization.
_DEFAULT_FILTER_TIMEOUT_SECONDS = 90.0

_DEFAULT_ANONYMIZER_MAX_TOKENS = 2048
_DEFAULT_ANONYMIZER_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Generic model-output JSON-envelope parser — used by every SOTA/local
# structured-output parser in this module and in extractor.py.
# ---------------------------------------------------------------------------

_JSON_ENVELOPE_KEYS = frozenset(
    (
        # Local extractor / SOTA enrichment / procedural envelopes
        "facts",
        "entities",
        "relations",
        "new_entity_bindings",
        "summary",
        # Anonymizer envelope — `{"mapping": {...}}`
        "mapping",
        # Plausibility drop-set envelope — `{"drop": [<idx>...]}` plus
        # tolerated aliases the parser accepts.
        "drop",
        "drop_indices",
        "indices",
        # SOTA enrichment delta envelope — `{"add": [...], "modify": [...],
        # "drop": [...], "bindings": {...}}`. ``drop`` is shared with
        # plausibility above. ``bindings`` overlaps `new_entity_bindings`
        # in role only — kept distinct because the new contract uses the
        # shorter name.
        "add",
        "modify",
        "bindings",
        # Predicate-normalization envelope (normalize_predicates):
        # single-stage output `{"clusters": [["predA", "predB"], ...]}`.
        "clusters",
    )
)


def _extract_json_block(text: str) -> str:
    """Extract a JSON object/array envelope from model output.

    Handles three real-world failure modes the model produces:

    1. **Code-fenced output**: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\``` — the
       fence is stripped before scanning.
    2. **Prose preamble with brace-quoted placeholder names**: SOTA
       sometimes narrates "I'll introduce ``{Topic_1}`` for the degree
       field" *before* emitting the JSON envelope.  The literal
       ``{Topic_1}`` is not parseable JSON (unquoted key) — a naive
       "first ``{``" parser would fail there and miss the real envelope
       further down.
    3. **Truncated outer envelope**: a model cut at ``max_tokens`` mid-
       string emits a valid *inner* sub-object ``{"name": "x", ...}``
       even though the outer envelope never closed.  Returning that
       inner object would silently produce an empty graph downstream.

    Algorithm: walk every ``{`` / ``[`` position in the text and try
    ``raw_decode`` from each.  Accept the first candidate that yields
    either a non-empty list, or a dict with at least one envelope key
    (:data:`_JSON_ENVELOPE_KEYS`).  Inner sub-objects without envelope
    keys are skipped so truncation surfaces as a parse failure rather
    than a silently empty graph.  Preamble noise (``{Topic_1}``) is
    skipped automatically because it raises ``JSONDecodeError``
    immediately.

    Error messages distinguish three fail modes:
    a. ``saw_decode_success_no_envelope=True`` → JSON parsed but no
       envelope keys → outer-envelope truncation if ``src`` does not end
       on a closing brace/bracket, otherwise (response looks complete)
       invalid JSON inside a string value — e.g. an unescaped control
       character (illegal per RFC 8259) — is reported as the likely
       cause instead.
    b. ``last_exc is not None`` → some ``{`` / ``[`` candidates existed
       but none parsed → likely all garbage / prose / corruption.
    c. No candidates at all → no JSON in the response.
    """
    # Strip markdown code-fence wrappers if present.
    src = text
    for marker in ("```json", "```"):
        if marker in src:
            start = src.index(marker) + len(marker)
            closing = src.find("```", start)
            if closing != -1:
                src = src[start:closing].strip()
                break

    decoder = json.JSONDecoder()
    last_exc: json.JSONDecodeError | None = None
    last_offset = -1
    saw_decode_success_no_envelope = False
    src_len = len(src)
    pos = 0
    while pos < src_len:
        next_brace = src.find("{", pos)
        next_bracket = src.find("[", pos)
        if next_brace < 0 and next_bracket < 0:
            break
        if next_brace < 0:
            candidate = next_bracket
        elif next_bracket < 0:
            candidate = next_brace
        else:
            candidate = min(next_brace, next_bracket)
        last_offset = candidate
        try:
            value, end = decoder.raw_decode(src, candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
            # Skip past the failing opener and keep looking.
            pos = candidate + 1
            continue
        # raw_decode succeeded — check whether this is the envelope or
        # narrative noise / a truncated inner sub-object.
        # Acceptance rules — both designed to surface truncation rather
        # than silently consume a sub-structure of a truncated envelope:
        #   • dict: must carry an envelope key (entities / relations /
        #     facts / new_entity_bindings / summary / mapping / ...).
        #     Rejects naked entity / relation dicts that survive a
        #     truncated outer envelope.
        #   • list: must be empty (plausibility's "all dropped" output)
        #     OR have first element shaped like a fact (subject /
        #     predicate / object key present). Rejects an entity list
        #     that survives a truncated outer envelope — its elements
        #     have ``name`` / ``entity_type``, not ``subject``.
        if isinstance(value, dict):
            # Empty dict is unambiguously "no items" (drop-set-style
            # no-op delta), symmetric to the empty-list envelope below.
            # A truncated outer envelope cannot produce `{}` because the
            # model always emits at least one key before EOF or never
            # closes the outer brace at all.
            is_envelope = not value or bool(set(value.keys()) & _JSON_ENVELOPE_KEYS)
        elif isinstance(value, list):
            if not value:
                is_envelope = True
            else:
                first = value[0]
                if isinstance(first, dict):
                    is_envelope = "subject" in first or "predicate" in first or "object" in first
                elif isinstance(first, int) and not isinstance(first, bool):
                    # Bare integer array — plausibility's tolerated drop-set
                    # form (`[0, 2, 5]`).  Accepting any int-first list as
                    # an envelope is safe: extractor / enrichment / anon
                    # outputs that survive a truncated outer envelope have
                    # dict-shaped first elements (entity / relation / fact),
                    # not bare ints.
                    is_envelope = True
                else:
                    is_envelope = False
        else:
            is_envelope = False
        if is_envelope:
            return src[candidate:end]
        saw_decode_success_no_envelope = True
        pos = end

    if saw_decode_success_no_envelope:
        # Some JSON parsed cleanly but none had envelope keys.  Two
        # distinct root causes land here and must not be conflated:
        # (a) the outer envelope was truncated at ``max_tokens`` and
        # only inner sub-objects survived — ``src`` will not end on a
        # closing brace/bracket; (b) the response was complete (``src``
        # DOES end on a closing brace/bracket) but the outer envelope
        # still failed ``raw_decode`` for another reason — e.g. an
        # unescaped control character inside a string value (illegal
        # per RFC 8259) — and only an inner sub-object parsed cleanly.
        # Asserting truncation unconditionally misdiagnoses (b).
        truncated_looking = not src.rstrip().endswith(("}", "]"))
        if truncated_looking:
            likely_cause = f"outer envelope truncated at max_tokens (response length {src_len})"
        else:
            likely_cause = (
                "response looks complete (ends on a closing brace/bracket) but the "
                "outer envelope still failed to parse — check for an unescaped "
                "control character or other invalid JSON inside a string value"
            )
        last_error_note = (
            f" Last parse error before the accepted candidate: {last_exc.msg} "
            f"(offset {last_offset})."
            if last_exc is not None
            else ""
        )
        raise ValueError(
            "Parsed JSON values found in model output but none have "
            "envelope keys (entities/relations/facts/new_entity_bindings/"
            f"summary; non-empty list also accepted). Likely cause: {likely_cause}."
            f"{last_error_note} Bump extraction_max_tokens or reduce input chunk "
            "size if truncated."
        )
    if last_exc is not None:
        # Candidates existed but none parsed — could be prose-only output,
        # severely malformed JSON, or a single oversized envelope that hit
        # max_tokens before its first ``{`` even closed.
        raise ValueError(
            f"No parseable JSON in model output (last error at offset "
            f"{last_offset}: {last_exc.msg}; response length {src_len}). "
            f"Possible causes: max_tokens hit before envelope closed, "
            f"or model emitted prose without a JSON envelope."
        )
    raise ValueError("No JSON found in model output")


# ---------------------------------------------------------------------------
# Local-model anonymizer — the LLM call (A) wraps. Moved here in full from
# ``paramem.graph.extractor`` (was: ``extractor.py`` module functions).
# ---------------------------------------------------------------------------


def load_anonymization_prompt(
    prompts_dir: str | Path | None = None,
    filename: str = "anonymization.txt",
) -> str:
    """Single source of truth for the anonymization prompt.

    Both the local-model and cloud-extractor anonymization paths read through
    this helper so a `configs/prompts/anonymization.txt` override applies to
    both — no silent divergence.

    ``prompts_dir`` overrides the search directory (forwarded to
    :func:`~paramem.graph.prompts._load_prompt`); ``filename`` overrides the
    template name so a calibration operator can point at an alternate prompt
    file.  Both default to the production values, so production callers are
    unaffected.

    The prompt this function loads is external config — edit
    ``configs/prompts/anonymization.txt`` to tune; no code changes are
    needed.
    """
    return _load_prompt(filename, prompts_dir=prompts_dir, required=True)


def anonymize_with_local_model(
    graph: SessionGraph,
    model,
    tokenizer,
    *,
    scrub: set[str] | frozenset[str],
    transcript: str = "",
    max_tokens: int = _DEFAULT_ANONYMIZER_MAX_TOKENS,
    temperature: float = _DEFAULT_ANONYMIZER_TEMPERATURE,
    seed: int | None = None,
    prompts_dir: str | Path | None = None,
    prompt_filename: str = "anonymization.txt",
) -> tuple[dict | None, str, str]:
    """Identify the ``real_name -> placeholder`` mapping AND the
    model-authored anonymized transcript using the local model.

    Returns ``(mapping, anonymized_transcript, raw_output)``.  The model
    is shown ``graph.relations`` (rendered as facts), ``transcript``, and
    ``scrub`` (the operator's PII-vocabulary hints, rendered into the
    prompt's ``{scrub_categories}`` slot) and is the SOLE scope authority
    (no code-side entity-type gate): it decides which real values —
    names AND structured values (phone, email, …), regardless of owning
    entity type — are in scope, and returns BOTH the mapping AND its own
    rewrite of ``transcript`` with those values placeholdered
    (``anonymized_transcript``).  The SCRIPT still builds the anonymized
    FACT array deterministically from ``graph.relations`` and ``mapping``
    (see :func:`anonymize_for_cloud`) — facts are never taken from the
    model's response — so a fact can never be lost, reworded, or dropped
    by the anonymizer, and a placeholder can never be glued into a
    predicate at this stage.

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

    ``prompts_dir`` and ``prompt_filename`` are forwarded to
    :func:`load_anonymization_prompt` so a calibration override actually
    reaches the model; both default to the production template.

    THE only production caller of this function is
    :func:`anonymize_for_cloud`'s step 2 — every path that anonymizes for
    cloud egress reaches the model through the one chain, never this
    function directly.
    """
    facts = [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in graph.relations
    ]

    anon_prompt = load_anonymization_prompt(prompts_dir=prompts_dir, filename=prompt_filename)
    anon_system = _load_prompt("anonymization_system.txt", prompts_dir=prompts_dir, required=True)
    # Sorted so prompt rendering is deterministic regardless of the
    # iteration order of the ``scrub`` set/frozenset the caller passed
    # (Python set order is not stable across processes) — load-bearing at
    # temperature=0.0, which is otherwise a strict determinism contract.
    prompt = anon_prompt.format(
        scrub_categories=", ".join(sorted(scrub)),
        facts_json=json.dumps(facts, indent=2),
        transcript=transcript or "(no transcript provided)",
    )
    messages = [
        {"role": "system", "content": anon_system},
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


# ---------------------------------------------------------------------------
# (A) — anonymize for cloud.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnonymizedPayload:
    """Result of :func:`anonymize_for_cloud` — the ONE fail-closed
    vocabulary for every cloud-egress anonymize call, replacing four
    ad-hoc shapes (``mapping is None``, ``("", {}, {})``, and two separate
    ``continue``s) that used to exist across the five migrated paths.

    ``status``:

    * ``"opted_out"`` — ``scrub`` was empty (operator opt-out).  No model
      call was made.  ``anon_transcript`` is ``transcript`` verbatim
      (sourced from the argument, never a model artifact); ``forward`` /
      ``reverse`` are ``{}``; ``anon_facts`` is the identity substitution
      of ``graph.relations`` (no-op, since ``forward`` is empty).
    * ``"failed"`` — fail-closed: the local anonymizer failed to parse or
      returned a missing/empty ``anonymized_transcript`` (step 3), OR the
      domain-scoped privacy guard fired (step 6).  ``forward`` /
      ``reverse`` / ``anon_facts`` are empty; ``anon_transcript`` is
      ``""``.  Callers must NEVER fall back to the original real-name
      transcript on this status.  ``failure`` distinguishes which of the
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
      validation (:func:`~paramem.graph.placeholders.
      _normalize_anonymization_mapping`) or by the node-key reconciliation
      above.  This is a classification/identity-match failure, not a
      parse failure — the anonymizer ran and returned a well-formed
      mapping; it just didn't survive onto this chunk's actual domain.

    This field exists because ``status == "failed"`` alone collapses two
    causes a caller may need to react to differently (see
    :func:`~paramem.training.graph_enrich.run_graph_enrichment`'s
    diagnostics), and
    ``rekey_dropped`` — a count, not a cause flag — is a lossy proxy: it
    is ``0`` both when the anonymizer never ran (``failure="parse"``) AND
    when it ran but every entry was dropped by shape validation before
    the reconciliation loop that increments it ever executed
    (``failure="guard"``).  Never reconstruct the cause from
    ``rekey_dropped``; read ``failure`` directly.
    """

    status: Literal["ok", "opted_out", "failed"]
    forward: dict[str, str]
    reverse: dict[str, str]
    anon_transcript: str
    anon_facts: list[dict]
    declared: frozenset[str]
    norm_stats: dict[str, int]
    rekey_dropped: int
    raw: str
    failure: Literal["parse", "guard"] | None = None


def anonymize_for_cloud(
    graph: SessionGraph,
    model,
    tokenizer,
    *,
    transcript: str,
    scrub: set[str] | frozenset[str],
    speaker_name: str | None = None,
    identity_domain: Iterable[str] | None = None,
    max_tokens: int = _DEFAULT_ANONYMIZER_MAX_TOKENS,
    seed: int | None = None,
    prompts_dir: str | Path | None = None,
    prompt_filename: str | None = None,
) -> AnonymizedPayload:
    """Anonymize a session's facts + transcript for cloud egress — THE one
    anonymize chain every cloud-bound path (session-tier extraction,
    graph-tier enrichment, chat egress, and their calibration harnesses)
    composes through.

    Serves both "facts but no transcript" (graph tier: ``transcript=""``)
    and "transcript but no facts" (chat egress: an ephemeral
    ``SessionGraph`` with no persisted relations) via the SAME signature —
    no flag, no branch.  ``graph.relations`` may be empty (``anon_facts``
    comes back ``[]``); ``transcript`` may be empty (``anon_transcript``
    comes back ``""`` on the opt-out path, sourced from the argument).

    Fixed sequence:

    1. ``scrub`` empty -> ``status="opted_out"``, empty tables,
       ``anon_transcript=transcript`` verbatim, ``anon_facts`` built via
       the identity (empty) mapping.
    2. :func:`anonymize_with_local_model` -> ``(llm_mapping,
       model_anon_transcript, raw)``.
    3. ``llm_mapping is None`` -> ``status="failed"``, empty tables,
       ``anon_transcript=""``.  Never falls back to the real-name
       transcript.
    4. :func:`~paramem.graph.placeholders._normalize_anonymization_mapping`
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
       non-empty AND the reconciled mapping came back empty AND
       ``graph.relations``' subject/object endpoints contain a
       non-speaker name -> ``status="failed"``.  The guard domain is
       deliberately derived from ``graph.relations`` — NOT from
       ``identity_domain`` — because the two differ: ``identity_domain``
       (the rekey domain) is a caller-supplied node list that may be
       trimmed to a size cap; the guard domain is the triples' actual
       subject/object endpoints, a subset of that list once trimming
       drops nodes whose only surviving edges fell outside the chunk.
       Fusing the two would silently reject a chunk whose surviving
       edges are all speaker-only just because its (larger, untrimmed)
       node list still lists a non-speaker node — a false-positive
       regression, not a privacy fix.  This coupling — ``identity_domain``
       gates BOTH step 5 and step 6 — is why a caller passing a domain
       must read this docstring: an ``identity_domain=None`` case never
       reconciles and never guards.
    7. :func:`~paramem.graph.placeholders._build_anonymization_mapping`
       (``speaker_name=speaker_name``) -> ``(forward, reverse)``.  THE
       speaker-value guard now applies on every path by construction — no
       code path is left that inverts an unfiltered forward map.
    8. :func:`~paramem.graph.placeholders._build_anon_facts`
       (``graph.relations``, ``forward``) -> ``anon_facts``.
    9. :func:`~paramem.graph.placeholders._declared_placeholder_tokens`
       (``reverse``) -> ``declared`` — the CORE declared vocabulary (no
       SOTA bindings yet at this stage; see :meth:`CloudScope.for_response`
       for the union used at deanon time).

    ``prompt_filename`` — ``None`` (default) uses
    :func:`anonymize_with_local_model`'s own production default
    (``"anonymization.txt"``); a calibration operator override is passed
    through only when explicitly supplied, never as an accidental
    ``None`` override of the production template.
    """
    if not scrub:
        return AnonymizedPayload(
            status="opted_out",
            forward={},
            reverse={},
            anon_transcript=transcript,
            anon_facts=_build_anon_facts(graph.relations, {}),
            declared=frozenset(),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
            failure=None,
        )

    anon_kwargs: dict = {}
    if prompt_filename is not None:
        anon_kwargs["prompt_filename"] = prompt_filename
    llm_mapping, model_anon_transcript, raw = anonymize_with_local_model(
        graph,
        model,
        tokenizer,
        transcript=transcript,
        scrub=scrub,
        max_tokens=max_tokens,
        seed=seed,
        prompts_dir=prompts_dir,
        **anon_kwargs,
    )
    if llm_mapping is None:
        return AnonymizedPayload(
            status="failed",
            forward={},
            reverse={},
            anon_transcript="",
            anon_facts=[],
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

        endpoint_names = {
            str(getattr(r, field, "")) for r in graph.relations for field in ("subject", "object")
        }
        endpoint_names = {n for n in endpoint_names if n and not is_speaker_id(n)}
        if reconciled_from_nonempty and endpoint_names and not mapping:
            return AnonymizedPayload(
                status="failed",
                forward={},
                reverse={},
                anon_transcript="",
                anon_facts=[],
                declared=frozenset(),
                norm_stats=norm_stats,
                rekey_dropped=rekey_dropped,
                raw=raw or "",
                failure="guard",
            )

    forward, reverse = _build_anonymization_mapping(mapping, speaker_name=speaker_name)
    anon_facts = _build_anon_facts(graph.relations, forward)
    declared = frozenset(_declared_placeholder_tokens(reverse))

    return AnonymizedPayload(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript=model_anon_transcript,
        anon_facts=anon_facts,
        declared=declared,
        norm_stats=norm_stats,
        rekey_dropped=rekey_dropped,
        raw=raw or "",
        failure=None,
    )


# ---------------------------------------------------------------------------
# The scope object — the only place `observed` is computed and the only
# place `_resolution_map` is called for a cloud round trip.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudScope:
    """The legality/resolution scope for ONE cloud round trip.

    Collapses what was mirrored five times: the construction of
    ``observed`` (the placeholder tokens actually shown to the cloud) and
    the two :func:`~paramem.graph.placeholders._resolution_map` calls
    (bindings-aware and bindings-free) built from it.
    """

    reverse: dict[str, str]
    sota_bindings: dict[str, str]
    observed: frozenset[str]
    resolution: dict[str, str]
    core_resolution: dict[str, str]
    declared: frozenset[str]

    @classmethod
    def for_response(
        cls,
        payload: AnonymizedPayload,
        *,
        sota_bindings: dict[str, str] | None,
        sent: Iterable[str],
    ) -> "CloudScope":
        """Build the scope for a cloud response against ``payload`` and
        the exact strings (``sent``) the cloud provider was actually
        shown.

        ``observed = {tok for tok in payload.declared if any(tok in s for
        s in sent)}`` — substring membership over the DECLARED
        vocabulary, never a shape scrape.  A shape scrape reads whatever
        placeholder-shaped surface the text happens to carry, which need
        not be the surface the table declares; any divergence between the
        two silently scopes CORE to nothing (every token reads as an
        orphan -> every delta rejected -> enrichment silently dead while
        the cycle still "succeeds").

        ``sota_bindings`` is normalized here with ``placeholder_side="key"``
        — the ONE call, not one per caller — and non-``str`` pairs are
        filtered as a side effect of that normalize.

        ``declared`` on the returned scope is the UNION of ``payload``'s
        CORE vocabulary and the normalized ``sota_bindings`` keys (the
        same union :func:`~paramem.graph.placeholders._apply_bindings`
        computes internally from its own ``reverse``/``sota_bindings``
        arguments) — used by :func:`deanonymize_response_text`'s
        fail-closed declared-token check.  It is deliberately NOT scoped
        to ``observed``: an unobserved declared token appearing in cloud
        prose must DROP the answer, not resolve — resolving it is
        precisely the privacy gap this design closes.  Do not "fix" this
        by intersecting with ``observed``.
        """
        raw_bindings = sota_bindings or {}
        normalized_bindings, _stats = _normalize_anonymization_mapping(
            raw_bindings, placeholder_side="key"
        )
        sent_tuple = tuple(sent)
        observed = frozenset(tok for tok in payload.declared if any(tok in s for s in sent_tuple))
        resolution = _resolution_map(payload.reverse, normalized_bindings, observed)
        core_resolution = _resolution_map(payload.reverse, {}, observed)
        declared = frozenset(_declared_placeholder_tokens(payload.reverse, normalized_bindings))
        return cls(
            reverse=payload.reverse,
            sota_bindings=normalized_bindings,
            observed=observed,
            resolution=resolution,
            core_resolution=core_resolution,
            declared=declared,
        )


# ---------------------------------------------------------------------------
# (B) — deanonymize from cloud.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeanonResult:
    """Result of :func:`deanonymize_facts`.

    ``verdict`` empty == accepted (``facts`` holds the substituted,
    already-partitioned survivors).  ``verdict`` non-empty == REJECT THE
    WHOLE DELTA — ``facts`` is ``[]`` and nothing was substituted; the
    caller decides the fallback (never a partial application).

    ``collisions`` is the totality gate's collision scan
    (:func:`~paramem.graph.placeholders._check_mapping_totality`'s second
    return value) — SOTA-binding keys that clash with the CORE reverse
    map or with the ``observed`` scope.  It is populated on BOTH exits
    (rejected and accepted): under an ``observed``-scoped call every
    collision is also folded into ``verdict``, but under the CORE-unscoped
    shape a collision is informational only and would otherwise be
    invisible to the caller.  Callers that keep diagnostics write it to
    ``graph.diagnostics["sota_binding_collisions"]`` themselves — this
    primitive mutates nothing.
    """

    facts: list[dict]
    verdict: list[str]
    collisions: list[str]
    predicate_dropped: list[dict]
    residual_dropped: list[dict]


def deanonymize_facts(
    scope: CloudScope,
    facts: list[dict],
) -> DeanonResult:
    """De-anonymize a cloud-returned fact delta — the single deanon exit
    gate for facts, run in fixed order:

    1. :func:`~paramem.graph.placeholders._check_mapping_totality` —
       unconditionally, as step 1.  A caller cannot skip the totality
       gate because it is inside this primitive, not something the
       caller opts into.
    2. A non-empty verdict rejects the WHOLE delta: returns
       ``DeanonResult(facts=[], verdict=verdict, collisions=..., [], [])``
       without substituting anything.  The caller decides the fallback
       (e.g. reverting to pre-enrichment local facts).
    3. :func:`~paramem.graph.placeholders._apply_bindings` — predicate
       invariant, substitute, residual sweep, fail-closed.

    Takes NO graph: the totality gate's two findings — ``verdict`` and
    ``collisions`` — come back on :class:`DeanonResult` as values.  A
    caller that keeps diagnostics writes them onto its own graph
    (``"sota_pending_orphans"`` / ``"sota_binding_collisions"``) right
    after this call; nothing here reaches into caller-owned state.
    """
    verdict, collisions = _check_mapping_totality(
        facts,
        scope.reverse,
        sota_bindings=scope.sota_bindings,
        observed=scope.observed,
    )
    if verdict:
        return DeanonResult(
            facts=[],
            verdict=verdict,
            collisions=collisions,
            predicate_dropped=[],
            residual_dropped=[],
        )

    kept, predicate_dropped, residual_dropped = _apply_bindings(
        facts, scope.reverse, scope.sota_bindings, scope.observed
    )
    return DeanonResult(
        facts=kept,
        verdict=[],
        collisions=collisions,
        predicate_dropped=predicate_dropped,
        residual_dropped=residual_dropped,
    )


def deanonymize_response_text(scope: CloudScope, text: str) -> str | None:
    """De-anonymize cloud-returned free text — the single deanon exit
    gate for prose (as opposed to fact dicts; see :func:`deanonymize_facts`).

    1. :func:`~paramem.graph.placeholders.deanonymize_text` against
       ``scope.resolution`` — the ``observed``-scoped map, NEVER a raw
       reverse map.  The signature makes the unsafe call unexpressible:
       there is no way to hand this function a bare ``reverse`` dict.
    2. :func:`~paramem.graph.placeholders._contains_declared_token`
       against ``scope.declared`` -> returns ``None``.  Always
       fail-closed.  Declared tokens are machine-minted (``Prefix_N``
       from our own table), so a false positive here is not credible.

    Returns the de-anonymized text, or ``None`` when a declared-but-
    unobserved (or otherwise unresolved) placeholder survives — the
    caller must treat ``None`` as a hard rejection of the whole response,
    never fall back to the raw cloud text.

    Note: this function does NOT run the undeclared-orphan shape backstop
    (:data:`~paramem.graph.placeholders.PLACEHOLDER_TOKEN_RE`) that
    :func:`~paramem.graph.placeholders._apply_bindings` runs for fact
    fields.  That backstop is out of scope for this round of unification
    (free conversational prose is a far wider surface for false positives
    than a structured fact field — e.g. ``GPT_4``, ``COVID_19`` are
    legitimate shape matches) and is deferred to a follow-up gated on a
    dedicated probe sweep.  The fact paths (session-tier extraction and
    graph-tier enrichment, via :func:`deanonymize_facts` ->
    ``_apply_bindings``) are unaffected — that sweep is unchanged.

    Callers using this function for something OTHER than free
    conversational prose (the chat-egress response) must independently
    justify the omission, since the "wide false-positive surface"
    rationale above is prose-specific and does not automatically
    transfer.  The graph-tier enrichment ``same_as`` arm
    (:func:`~paramem.graph.extractor._graph_enrich_with_sota`) is the one
    other production caller — see that function's docstring for why the
    omission is safe there too, for an UNRELATED reason (a downstream
    graph-membership guard, not a prose/false-positive argument).
    """
    out = deanonymize_text(text, scope.resolution)
    if _contains_declared_token(out, scope.declared):
        return None
    return out
