"""De-anonymize a cloud response — step (B) of the cloud round trip, plus
the scope object every response is checked and substituted against.

``paramem.cloud.placeholders`` is the primitive kit: mint, substitute,
normalize, invert, resolution map, placeholder-token scan, binding
collisions, apply-bindings — model-free, IO-free. This module is the
composed exit gate every cloud-returned
artifact (facts OR prose) passes through: :class:`CloudScope` (the one
legality/resolution scope for a response), :func:`deanonymize_facts` (the
fact-list exit), :func:`deanonymize_text` (the prose exit).

``_extract_json_block`` lives here (the generic model-output JSON-envelope
parser every cloud-response parser in :mod:`paramem.graph.extractor` and
:func:`~paramem.cloud.anonymize.anonymize_transcript` uses) rather than in
``paramem.graph.extractor`` because that module imports FROM this one — a
function-local import to dodge the reverse direction would split one
contract across two files for no reason other than import ordering.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paramem.cloud.placeholders import (
    _apply_bindings,
    _binding_collisions,
    _contains_declared_token,
    _declared_placeholder_tokens,
    _normalize_anonymization_mapping,
    _placeholder_tokens,
    _resolution_map,
    _substitute_whole_words,
)

if TYPE_CHECKING:
    from paramem.cloud.anonymize import AnonymizedContract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic model-output JSON-envelope parser — used by every cloud/local
# structured-output parser in this package and in
# ``paramem.graph.extractor``.
# ---------------------------------------------------------------------------

_JSON_ENVELOPE_KEYS = frozenset(
    (
        # Local extractor / cloud enrichment / procedural envelopes
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
        # cloud enrichment delta envelope — `{"add": [...], "modify": [...],
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
    2. **Prose preamble with brace-quoted placeholder names**: Cloud
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
# The scope object — the only place `observed` is computed and the only
# place `_resolution_map` is called for a cloud round trip.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudScope:
    """The legality/resolution scope for ONE cloud round trip.

    Collapses what was mirrored five times: the construction of
    ``observed`` (the placeholder tokens actually shown to the cloud) and
    the two :func:`~paramem.cloud.placeholders._resolution_map` calls
    (bindings-aware and bindings-free) built from it.

    ``core_resolution`` (the bindings-FREE map — CORE only, ``cloud_bindings={}``)
    has exactly ONE production consumer: a ``len()`` in the ``cloud_enrich``
    phase-trace diagnostic's rejection-branch ``"mapped_count"`` (the
    ``enrich`` stage, :mod:`paramem.graph.stage_enrich`) — everywhere else that
    needs a resolution map reads ``resolution`` (the bindings-aware one).
    Kept as a field rather than computed at that one call site: it is
    already produced by the SAME :func:`~paramem.cloud.placeholders.
    _resolution_map` call this classmethod makes for ``resolution``
    (``cloud_bindings={}`` is a cheap variant, not a second lookup), and
    inlining it at the one caller would recreate exactly the
    multiple-recomputation shape :func:`~paramem.cloud.deanonymize.
    deanonymize_facts`'s ``resolution=scope.resolution`` threading exists
    to avoid.
    """

    reverse: dict[str, str]
    cloud_bindings: dict[str, str]
    observed: frozenset[str]
    resolution: dict[str, str]
    core_resolution: dict[str, str]
    declared: frozenset[str]

    @classmethod
    def response(
        cls,
        contract: "AnonymizedContract",
        *,
        cloud_bindings: dict[str, str] | None,
        sent: Iterable[str],
    ) -> "CloudScope":
        """Build the scope for a cloud response against ``contract`` and
        the exact strings (``sent``) the cloud provider was actually
        shown.

        ``observed = {tok for tok in contract.declared if any(tok in s for
        s in sent)}`` — substring membership over the DECLARED
        vocabulary, never a shape scrape.  A shape scrape reads whatever
        placeholder-shaped surface the text happens to carry, which need
        not be the surface the table declares; any divergence between the
        two silently scopes CORE to nothing (every token reads as an
        orphan -> every delta rejected -> enrichment silently dead while
        the cycle still "succeeds").

        ``cloud_bindings`` is normalized here with ``placeholder_side="key"``
        — the ONE call, not one per caller — and non-``str`` pairs are
        filtered as a side effect of that normalize.

        **Binding-value pruning (2026-07-22 cloud-admission redesign).**
        After normalizing, any binding whose own VALUE still carries an
        unresolvable placeholder token (the ``{"Role_1": "Senior Engineer
        at Org_9"}`` case, where ``Org_9`` is never declared anywhere) is
        dropped from ``cloud_bindings`` entirely — ONE pass, not a
        fixpoint: a value that itself references another cloud-minted
        binding chain (``Role_1``'s value naming ``Org_9``, whose own
        value in turn names a THIRD unresolved token) only has its
        outermost link checked here.  Deliberate: this scope has already
        computed ``resolution``/``declared`` from ``cloud_bindings``
        before any transitive chain could be walked, and a fixpoint loop
        over a cloud-controlled dict is an unbounded-iteration surface for
        no observed benefit (the prompt's own contract mints one binding
        per new entity, not chains of them).  A pruned binding is simply
        absent from ``resolution``, so any fact whose subject/object
        referenced it becomes unresolvable and is dropped downstream by
        the ordinary per-fact rule (:func:`~paramem.graph.extractor.
        _apply_enrichment_delta`'s orphan check on ``add``/``modify``, or
        :func:`~paramem.cloud.placeholders._apply_bindings`'s residual
        sweep) — this replaces the fatal binding-value scan that used to
        live in the now-retired ``_check_mapping_totality`` (folded into
        :func:`~paramem.cloud.placeholders._binding_collisions`, which no
        longer scans binding values at all — that job moved here).

        ``declared`` on the returned scope is the UNION of ``contract``'s
        CORE vocabulary and the normalized ``cloud_bindings`` keys (the
        same union :func:`~paramem.cloud.placeholders._apply_bindings`
        computes internally from its own ``reverse``/``cloud_bindings``
        arguments) — used by :func:`deanonymize_text`'s fail-closed
        declared-token check.  It is deliberately NOT scoped to
        ``observed``: an unobserved declared token appearing in cloud
        prose must DROP the answer, not resolve — resolving it is
        precisely the privacy gap this design closes.  Do not "fix" this
        by intersecting with ``observed``.

        Args:
            contract: The :func:`~paramem.cloud.anonymize.anonymize`
                result this response's round trip is scoped against —
                ``.declared`` and ``.reverse`` are read.  The type import
                is ``TYPE_CHECKING``-only (never at runtime): this
                function reads structurally, so nothing here creates a
                runtime import cycle with ``anonymize.py`` (which DOES
                import this module at runtime, for
                :data:`_extract_json_block`).
        """
        raw_bindings = cloud_bindings or {}
        normalized_bindings, _stats = _normalize_anonymization_mapping(
            raw_bindings, placeholder_side="key"
        )
        sent_tuple = tuple(sent)
        observed = frozenset(tok for tok in contract.declared if any(tok in s for s in sent_tuple))

        # Binding-value pruning — see docstring. A binding whose VALUE
        # still carries a placeholder token unresolvable against the
        # CORE-plus-bindings domain is dropped entirely (ONE pass).
        if normalized_bindings:
            _prune_resolvable = set(
                _resolution_map(contract.reverse, normalized_bindings, observed)
            )
            _pruned_bindings: dict[str, str] = {}
            for _key, _value in normalized_bindings.items():
                if _placeholder_tokens(_value) - _prune_resolvable:
                    logger.warning(
                        "cloud binding %r pruned — its value %r carries an "
                        "unresolvable placeholder token",
                        _key,
                        _value,
                    )
                    continue
                _pruned_bindings[_key] = _value
            normalized_bindings = _pruned_bindings

        resolution = _resolution_map(contract.reverse, normalized_bindings, observed)
        core_resolution = _resolution_map(contract.reverse, {}, observed)
        declared = frozenset(_declared_placeholder_tokens(contract.reverse, normalized_bindings))
        return cls(
            reverse=contract.reverse,
            cloud_bindings=normalized_bindings,
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

    ``facts`` holds the substituted, already-partitioned survivors —
    ``deanonymize_facts`` always substitutes now (2026-07-22
    cloud-admission redesign retired the whole-delta accept/reject
    ``verdict`` this dataclass used to carry): per-triple resolvability is
    decided earlier, at the point cloud's delta is applied
    (:func:`~paramem.graph.extractor._apply_enrichment_delta`), and the
    fail-closed residual sweep in :func:`~paramem.cloud.placeholders.
    _apply_bindings` (surfaced here as ``predicate_dropped`` /
    ``residual_dropped``) is what still sheds an individual fact this
    call.

    ``collisions`` is :func:`~paramem.cloud.placeholders._binding_collisions`'s
    result — cloud-binding keys that clash with the CORE reverse map or
    with the ``observed`` scope.  ALWAYS informational: a binding for a
    token cloud was SHOWN is inert under CORE-LAST precedence
    (:func:`~paramem.cloud.placeholders._resolution_map`), never a reason
    to drop anything.  Callers that keep diagnostics write it to their own
    diagnostics store themselves — this primitive mutates nothing.
    """

    facts: list[dict]
    collisions: list[str]
    predicate_dropped: list[dict]
    residual_dropped: list[dict]


def deanonymize_facts(
    scope: CloudScope,
    facts: list[dict],
) -> DeanonResult:
    """De-anonymize a cloud-returned fact delta — the single deanon exit
    gate for facts.

    Always substitutes via :func:`~paramem.cloud.placeholders.
    _apply_bindings` (predicate invariant, substitute, residual sweep,
    fail-closed) — there is no whole-delta accept/reject step here any
    more.  :func:`~paramem.cloud.placeholders._binding_collisions` still
    runs, purely for its ``collisions`` diagnostic; per-triple
    resolvability (dropping an unresolvable ``add``, reverting an
    unresolvable ``modify``) is decided earlier, by
    :func:`~paramem.graph.extractor._apply_enrichment_delta`, before this
    function ever sees the fact list.

    Takes NO graph/diagnostics sink: ``collisions`` comes back on
    :class:`DeanonResult` as a value.  A caller that keeps diagnostics
    writes it onto its own store right after this call; nothing here
    reaches into caller-owned state.
    """
    collisions = _binding_collisions(
        scope.reverse,
        cloud_bindings=scope.cloud_bindings,
        observed=scope.observed,
    )
    # ``resolution=scope.resolution``: the scope already computed the
    # resolution map once (in :meth:`CloudScope.response`) — passing it
    # through here avoids :func:`~paramem.cloud.placeholders._apply_bindings`
    # rebuilding it fresh from the same ``reverse``/``cloud_bindings``/
    # ``observed`` triple.
    kept, predicate_dropped, residual_dropped = _apply_bindings(
        facts, scope.reverse, scope.cloud_bindings, scope.observed, resolution=scope.resolution
    )
    return DeanonResult(
        facts=kept,
        collisions=collisions,
        predicate_dropped=predicate_dropped,
        residual_dropped=residual_dropped,
    )


def deanonymize_text(scope: CloudScope, text: str) -> str | None:
    """De-anonymize cloud-returned free text — the single deanon exit
    gate for prose (as opposed to fact dicts; see :func:`deanonymize_facts`).

    1. :func:`~paramem.cloud.placeholders._substitute_whole_words` against
       ``scope.resolution`` — the ``observed``-scoped map, NEVER a raw
       reverse map.  The signature makes the unsafe call unexpressible:
       there is no way to hand this function a bare ``reverse`` dict.
       (This step used to be a dedicated ``deanonymize_text`` primitive in
       the placeholder kit; that primitive was a pure pass-through null
       guard already subsumed by ``_substitute_whole_words``'s own
       ``if not mapping or not text`` guard, so it collapsed into a
       direct call here rather than surviving as a second name for the
       same one-line body — see ``paramem.cloud.placeholders`` module
       docstring.)
    2. :func:`~paramem.cloud.placeholders._contains_declared_token`
       against ``scope.declared`` -> returns ``None``.  Always
       fail-closed.  Declared tokens are machine-minted (``Prefix_N``
       from our own table), so a false positive here is not credible.

    Returns the de-anonymized text, or ``None`` when a declared-but-
    unobserved (or otherwise unresolved) placeholder survives — the
    caller must treat ``None`` as a hard rejection of the whole response,
    never fall back to the raw cloud text.

    Note: this function does NOT run the undeclared-orphan shape backstop
    (:data:`~paramem.cloud.placeholders.PLACEHOLDER_TOKEN_RE`) that
    :func:`~paramem.cloud.placeholders._apply_bindings` runs for fact
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
    (:func:`~paramem.graph.extractor.request_graph_enrichment`) is the one
    other production caller — see that function's docstring for why the
    omission is safe there too, for an UNRELATED reason (a downstream
    graph-membership guard, not a prose/false-positive argument).
    """
    out = _substitute_whole_words(text, scope.resolution)
    if _contains_declared_token(out, scope.declared):
        return None
    return out
