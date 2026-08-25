"""Anonymize a fact list / transcript for cloud egress — step (A) of the
cloud round trip.

Serves every cloud-bound path (session-tier extraction, graph-tier
enrichment, chat egress, and their calibration harnesses) through
:func:`anonymize` — the chain around the tagger-backed scan
(:mod:`paramem.cloud.span_tagger`, via
:func:`~paramem.cloud.anonymize_steps.scan_values`) and the anonymizer's
one remaining local-model call, the ANCHOR self-introduction question
(:func:`~paramem.cloud.anonymize_steps.ask_speaker_anchor`). Every
placeholder is minted in code
(:func:`~paramem.cloud.placeholders.build_forward_table`) and the
anonymized transcript is produced by the same exact, case-sensitive
substitution primitive that already produces the anonymized facts
(:func:`~paramem.cloud.placeholders._substitute_whole_words`) — there is
no model-authored rewrite to verify.

Interface narrowing (2026-07-21): :func:`anonymize` takes
``facts: list[dict]`` — never a ``SessionGraph`` or ``Relation``.  A
``SessionGraph`` was a CARRIER on this boundary, not an artifact: the
pre-narrowing ``anonymize_for_cloud(graph, ...)`` touched ``graph`` only to
(a) render ``graph.relations`` into the prompt's fact payload and (b)
harvest subject/object surfaces for the identity-reconciliation guard —
both are plain projections of ``Iterable[Relation]`` a caller can render
once, caller-side, in ``paramem/graph/``.

Likewise this module never loads its own prompt file: this package must
import nothing from ``paramem.graph``, and the prompt-loading + calibration-override +
provenance-recording chokepoint (:func:`~paramem.graph.prompts._load_prompt`)
lives there.  The caller (inside ``paramem/graph/``, already holding the
active ``phase_trace`` scope this call's provenance records onto) composes
:class:`AnonymizerPrompts` via
:func:`~paramem.graph.anonymizer_prompts.load_anonymizer_prompts` and
passes it in — data, not a loader capability, so the layering holds
without an injected callable.

:class:`AnonymizerPrompts` is declared HERE (pure data — no IO, no
``paramem.graph`` import) so that this module's own signatures can name
the type without reaching into ``paramem.graph``.  No import cycle
results: ``paramem.graph.
anonymizer_prompts`` imports ``AnonymizerPrompts`` FROM this module (one
direction, matching the existing overall rule that ``paramem.graph``
depends on ``paramem.cloud``, never the reverse) — this module does not,
and must not, import ``paramem.graph.anonymizer_prompts`` or anything else
under ``paramem.graph``.

There is no fact-boundary slicing and no per-category local call: the
payload partition now lives in the tagger, in the model's own splitter/
subword units (:mod:`paramem.cloud.span_tagger`), and the only
envelope-bearing local call left is the ANCHOR, whose input is the
history-plus-transcript evidence region, never the fact block. A single
:func:`anonymize` call therefore costs at most one local ``generate()``
call, not one per configured category.

Dynamic VRAM clamp (owner-approved 2026-07-28, live-fold evidence: a
packer-correct 8,192-token call still faulted "device not ready" at 1,191
MiB free): the configured ``token_envelope`` is the operator CEILING, not
a guarantee of what live free VRAM can support at call time — free VRAM
varies within one fold.  :func:`anonymize` measures free VRAM ONCE at its
own entry, via :func:`~paramem.utils.vram_guard.effective_token_envelope`,
and threads the resulting effective (possibly smaller) envelope to the
ANCHOR call — the only envelope-bearing call left; the tagger is CPU-only
and takes no envelope.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from paramem.cloud.anonymize_steps import ScanResult, ask_speaker_anchor, scan_values
from paramem.cloud.placeholders import (
    _MAX_MAPPING_TEXT_CHARS,
    _applied_whole_word_keys,
    _declared_placeholder_tokens,
    _substitute_whole_words,
    build_forward_table,
    invert_forward_mapping,
    placeholder_prefix,
)
from paramem.cloud.span_tagger import TaggedSpan, TaggerUnavailable
from paramem.config.taxonomy import ScrubCategory, entity_type_to_prefix
from paramem.utils.identity import canonical, is_speaker_id
from paramem.utils.tokens import ANONYMIZE_ENVELOPE_TOKENS
from paramem.utils.turn_markers import split_marker
from paramem.utils.vram_guard import effective_token_envelope

logger = logging.getLogger(__name__)

# Total tokens (prompt + output) the local ANCHOR call may occupy. 8192 is
# the sequence length vram.vram_cache_headroom_gib: 1.5 was booked against
# (configs/server.yaml.example — "single-sequence 8192-token Mistral KV cache
# is ~1 GiB; 0.5 GiB margin for activations"). Reads
# paramem.utils.tokens.ANONYMIZE_ENVELOPE_TOKENS — the one executable home
# for the literal, since paramem.graph.document_chunker and
# paramem.server.session_buffer also need this value and must stay
# importable without the cloud package (see paramem/utils/tokens.py).
_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE: int = ANONYMIZE_ENVELOPE_TOKENS


@dataclass(frozen=True)
class AnonymizerPrompts:
    """Composed, call-shape-ready ANCHOR prompt sections for one
    :func:`anonymize` run — pure data, no IO. Constructed by
    :func:`~paramem.graph.anonymizer_prompts.load_anonymizer_prompts` and
    nowhere else.

    The ANCHOR call is the only local-model call left in the chain — the
    scan is tagger-backed (no prompt) and the transcript/facts rewrite is
    code-side substitution (no prompt).

    Attributes:
        anchor_system: The ``ANCHOR-SYSTEM`` section, category-independent.
        anchor: The ``ANCHOR`` section (``{speaker_id}``, ``{values}``,
            ``{text}`` all deferred).
    """

    anchor_system: str
    anchor: str


@dataclass(frozen=True)
class TagPayload:
    """The two derivations :func:`_assemble_payload` produces from one set
    of inputs — the tagger's payload, and the ANCHOR call's own evidence.

    Attributes:
        tag_text: Marker-FREE, newline-joined: every history line, every
            transcript line (each with its ``[role] `` marker stripped via
            :func:`~paramem.utils.turn_markers.split_marker`), then one
            line per fact — that fact's ``subject`` and ``object`` joined
            verbatim by a single space, unquoted, unescaped (see
            :func:`_render_fact_lines`) — the ONE text handed to
            :func:`~paramem.cloud.anonymize_steps.scan_values`. This is
            load-bearing: the tagger must read exactly the strings
            :func:`~paramem.cloud.placeholders.insert_placeholders` later
            substitutes over (``subject``/``object`` only — ``predicate``
            is never a substitution target and carries nothing here). A
            rendering that quotes or escapes a fact value (the prior
            ``json.dumps(facts, ...)`` render did, for any value
            containing ``"`` or ``\\``) would make the tagger tag the
            ESCAPED surface — a string that never equals the real value
            the consumer substitutes, so the real value would egress
            unsubstituted. Its coordinates are the only offset space the
            arc uses.
        anchor_range: ``[start, end)`` of the transcript region inside
            ``tag_text`` — contiguous by construction, since the
            transcript lines are joined immediately after the history
            lines and immediately before the fact block.
        anchor_evidence: The join of the history + transcript lines
            EXACTLY as received — marker-BEARING, no fact block. The
            ANCHOR prompt is calibrated on marker-bearing text, so this is
            what the ANCHOR call is shown; it is input concatenation, not
            a rendering — no marker is stripped, re-added, or reformatted.
    """

    tag_text: str
    anchor_range: tuple[int, int]
    anchor_evidence: str


def _render_fact_lines(facts: list[dict]) -> str:
    """Render *facts* into the tagger's own view of them: one line per
    fact, that fact's ``subject`` and ``object`` joined verbatim by a
    single space — no quoting, no escaping.

    This is the SAME two fields, read the SAME way
    (``str(f.get("subject", ""))`` / ``str(f.get("object", ""))``), that
    :func:`~paramem.cloud.placeholders.insert_placeholders` later
    substitutes through the forward table this scan builds — the tagger's
    view is therefore a literal superset of every substitution surface.
    ``predicate``/``relation_type``/``confidence``/``speaker_id`` are
    never a substitution target (see
    :func:`~paramem.cloud.placeholders.insert_placeholders`'s docstring)
    and are deliberately excluded here: including them would tag content
    the consumer never rewrites, growing ``scan_dropped``/``inert_dropped``
    for no substitution benefit.
    """
    return "\n".join(f"{str(f.get('subject', ''))} {str(f.get('object', ''))}" for f in facts)


def _assemble_payload(
    history_lines: Sequence[str], transcript: str, facts: list[dict]
) -> TagPayload:
    """Build the tagger payload and the ANCHOR evidence text from one set
    of inputs — the ONE assembler producing both
    :class:`TagPayload` derivations.

    ``history_lines`` and ``transcript`` both arrive marker-bearing: each
    entry of *history_lines* is one turn already rendered via
    :func:`~paramem.utils.turn_markers.format_turn`; *transcript* is the
    session tier's own multi-line, ``\\n``-joined marker-bearing transcript
    (:meth:`~paramem.server.session_buffer.SessionBuffer._format_turns`),
    or a synthetic single line at chat egress, or ``""`` at the graph
    tier. Each line's role marker is removed via
    :func:`~paramem.utils.turn_markers.split_marker` — the ONE site above
    :func:`~paramem.cloud.anonymize_steps.scan_values` that changes offset
    space, and the only use of ``split_marker`` in the arc.

    ``tag_text`` orders history, then transcript, then the rendered fact
    lines (:func:`_render_fact_lines` — one line per fact, ``subject``
    and ``object`` verbatim, the SAME strings
    :func:`~paramem.cloud.placeholders.insert_placeholders` later
    substitutes), so the transcript region is contiguous and recorded
    exactly as ``anchor_range``. ``anchor_evidence`` is a separate join of
    the SAME lines, marker-bearing and with no fact block — it is never
    derived from ``tag_text`` (which has already had every marker
    stripped).
    """
    stripped_history = [split_marker(line)[1] for line in history_lines]
    transcript_lines = transcript.split("\n") if transcript else []
    stripped_transcript = [split_marker(line)[1] for line in transcript_lines]

    transcript_start = sum(len(line) + 1 for line in stripped_history)
    transcript_text = "\n".join(stripped_transcript)
    transcript_end = transcript_start + len(transcript_text)

    fact_lines = _render_fact_lines(facts)
    tag_text = "\n".join([*stripped_history, *stripped_transcript, fact_lines])
    anchor_evidence = "\n".join([*history_lines, *transcript_lines])

    return TagPayload(
        tag_text=tag_text,
        anchor_range=(transcript_start, transcript_end),
        anchor_evidence=anchor_evidence,
    )


def _anchor_candidates(
    scans: Sequence[ScanResult],
    spans: Sequence[TaggedSpan],
    anchor_range: tuple[int, int],
    person_prefix: str,
) -> tuple[str, ...]:
    """The person-row surfaces whose tagged span lies inside
    *anchor_range*, in first-appearance order — the closed candidate
    domain :func:`~paramem.cloud.anonymize_steps.ask_speaker_anchor` is
    shown and checks its answer against.

    ``scans`` supplies the VERIFIED surfaces (speaker-id and
    non-whole-word surfaces already dropped) for the category whose
    ``prefix`` equals *person_prefix*; ``spans`` supplies the offsets — a
    scan result alone carries no position, and a raw span alone carries no
    verification. A span is a candidate only when it is wholly contained
    in ``[anchor_range[0], anchor_range[1])`` — the transcript region
    :func:`_assemble_payload` recorded — so a value the tagger found only
    in history or only in the fact block is never offered to the ANCHOR
    call.
    """
    person_values: set[str] = set()
    for scan in scans:
        if scan.category.prefix == person_prefix:
            person_values = set(scan.values)
            break
    if not person_values:
        return ()

    start, end = anchor_range
    seen: set[str] = set()
    candidates: list[str] = []
    for span in spans:
        if span.text not in person_values:
            continue
        if span.start < start or span.end > end:
            continue
        if span.text in seen:
            continue
        seen.add(span.text)
        candidates.append(span.text)
    return tuple(candidates)


def _render_scan_raw(spans: Sequence[TaggedSpan], anchor_raw: str) -> str:
    """THE named ``contract.raw`` derivation: the tagged span list
    (``label``/``text``/``score``/``start``/``end``) plus the ANCHOR
    call's own raw text, rendered as one JSON object.

    On a tagger failure ``anonymize()`` never calls this — ``raw`` is the
    tagger's own refusal message instead (see :func:`anonymize`'s
    ``TaggerUnavailable`` catch).

    ``ensure_ascii=False``: a span whose ``text`` was correctly tagged
    verbatim (see ``TagPayload.tag_text``'s docstring) must read verbatim
    here too — the default (``ensure_ascii=True``) would escape every
    non-ASCII character, making a correctly-tagged span look identical to
    the historical escaped-surface defect this raw record is meant to let
    an operator rule out.
    """
    return json.dumps(
        {
            "spans": [
                {
                    "label": s.label,
                    "text": s.text,
                    "score": s.score,
                    "start": s.start,
                    "end": s.end,
                }
                for s in spans
            ],
            "anchor": anchor_raw,
        },
        ensure_ascii=False,
    )


def _dropped_inert_entry(real: str, placeholder: str) -> dict:
    """Build one ``scan_dropped_entries`` record for a forward-table key
    pruned by :func:`anonymize`'s inert-key pass: ``side="table"``,
    ``reason="inert"`` — distinguishing a key dropped because it never
    substituted anywhere in the payload from a scan-time drop
    (``side="scan"``, built by
    :func:`~paramem.cloud.anonymize_steps._dropped_scan_entry`).

    ``category`` is best-effort: *placeholder*'s own minted prefix
    (:func:`~paramem.cloud.placeholders.placeholder_prefix`), or ``""``
    when *placeholder* is not shape-matched (a speaker-anchor fold's
    placeholder is the well-shaped ``speaker{N}`` token itself, which
    never matches :data:`~paramem.cloud.placeholders.PLACEHOLDER_SHAPE_RE`)
    — the same truncation cap
    (:data:`~paramem.cloud.placeholders._MAX_MAPPING_TEXT_CHARS`) the scan
    path already uses for ``text``.
    """
    return {
        "category": placeholder_prefix(placeholder) or "",
        "side": "table",
        "text": real[:_MAX_MAPPING_TEXT_CHARS],
        "reason": "inert",
    }


@dataclass(frozen=True)
class AnonymizedContract:
    """Result of :func:`anonymize` — the ONE fail-closed vocabulary for
    every cloud-egress anonymize call.

    ``status``:

    * ``"opted_out"`` — *categories* was empty (an empty operator
      ``sanitization.scrub`` resolves to zero categories — see
      :func:`opted_out_contract`).  No tagger call, no model call.
      ``anon_transcript`` is ``transcript`` verbatim (sourced from the
      argument, never a derived artifact); ``forward`` / ``reverse`` are
      ``{}``; ``facts`` is the input ``facts`` verbatim (identity
      substitution downstream via
      :func:`~paramem.cloud.placeholders.insert_placeholders` over this
      empty ``forward``).
    * ``"failed"`` — fail-closed (see ``failure`` below).  ``forward`` /
      ``reverse`` / ``facts`` are empty; ``anon_transcript`` is ``""``.
      Callers must NEVER fall back to the original real-name transcript on
      this status, and must NOT derive an anonymized fact array from this
      contract.  Diagnostics (``call_tokens`` / ``scan_dropped`` /
      ``scan_dropped_entries`` / ``rekey_dropped``) are NOT zeroed here —
      :func:`failed_contract` carries the real accumulation up to the
      point the terminal fired, so a failed run's calibration artifact can
      still say WHY it failed.
    * ``"ok"`` — the scan ran, the domain guard did not fire, and the
      table was built.  ``forward`` / ``reverse`` may still be empty — a
      legitimate verdict ("ran, found nothing in scope"), not a failure;
      egress proceeds.  ``facts`` is the input ``facts`` verbatim.

    ``failure`` — ``None`` except when ``status == "failed"``, where it is
    always one of:

    * ``"guard"`` — the domain-scoped fail-closed guard fired: the scan
      named something, but nothing survived identity reconciliation onto
      this call's actual domain.
    * ``"tagger"`` — the span tagger is unavailable, or its model call
      raised (:class:`~paramem.cloud.span_tagger.TaggerUnavailable`,
      caught here and only here).

    ``reverse`` is the de-anonymization key: it must NEVER egress. The
    field that DOES egress is ``forward`` (used to placeholder outbound
    facts/transcript) plus ``anon_transcript``/``declared`` — naming this
    type ``AnonymizedContract`` rather than ``...Payload`` makes that
    asymmetry a property of the type, not just a comment on one field.

    ``facts`` is the (real-name, un-substituted) fact array cleared for
    egress — every production reader derives the anonymized array on
    demand via :func:`~paramem.cloud.placeholders.insert_placeholders`
    over ``facts`` and ``forward``, never storing the substituted array
    here.

    ``model_calls`` — local ``generate()`` calls actually issued: the
    ANCHOR call, and only the ANCHOR call (0 or 1).

    ``call_tokens`` — one record per ``generate()`` call actually issued,
    matching ``model_calls`` in length: ``{"label", "prompt_tokens",
    "output_tokens"}`` — the per-call telemetry
    :func:`~paramem.cloud.anonymize_steps._generate` measures, surfaced
    verbatim by
    :func:`~paramem.server.calibrate.dispatch_anonymize_facts`'s ``parsed``
    block.

    ``tagger_windows`` — :attr:`~paramem.cloud.span_tagger.TagResult.windows`,
    the number of tagger model calls the scan issued.

    ``scan_dropped`` / ``scan_dropped_entries`` are the mapping-quality
    signal: per-entry records ``{category, side: "scan"|"table",
    text: <truncated>, reason: "speaker_id" | "not_whole_word" |
    "contained" | "inert"}``, accumulated across every category — see
    :func:`~paramem.cloud.anonymize_steps.ScanResult`'s docstring for the
    ``side="scan"`` reasons. ``scan_dropped`` (the int) counts ONLY the
    ``side="scan"`` entries — the scan step's own drops, before a forward
    table even exists; it can be smaller than ``len(scan_dropped_entries)``
    once the ``side="table"``/``reason="inert"`` entries below are
    appended.

    ``inert_dropped`` is the count of forward-table keys :func:`anonymize`
    pruned because they substitute nothing anywhere in the payload — a
    forward key that matches no text is never a real substitution target;
    keeping it would leave a ``reverse`` entry a cloud reply could pull a
    real (possibly partial) value back through, without that value ever
    having actually been scrubbed from anything egressed. Applied
    uniformly, including a speaker-anchor fold's own placeholder: an
    anchor entry that never occurs verbatim in the payload is pruned like
    any other key, never carved out. The same ``side="table"``/
    ``reason="inert"`` entries land in ``scan_dropped_entries`` (see
    :func:`_dropped_inert_entry`), so a single diagnostic list carries
    every one of the four PER-ENTRY drop reasons (``"speaker_id"``,
    ``"not_whole_word"``, ``"contained"``, ``"inert"``). Two further
    reasons a forward-table key never makes it to ``reverse`` carry no
    entry here: the domain re-key drop (counted only in
    ``rekey_dropped``) and the speaker-anchor fold (a value folded onto
    ``speaker_id`` is excluded from ``reverse`` by design — see
    :func:`~paramem.cloud.placeholders.build_forward_table`'s docstring —
    never a drop at all).
    """

    status: Literal["ok", "opted_out", "failed"]
    forward: dict[str, str]
    reverse: dict[str, str]
    anon_transcript: str
    declared: frozenset[str]
    rekey_dropped: int
    raw: str
    failure: Literal["guard", "tagger"] | None = None
    facts: list[dict] = field(default_factory=list)
    tagger_windows: int = 0
    model_calls: int = 0
    call_tokens: tuple[dict, ...] = ()
    scan_dropped: int = 0
    scan_dropped_entries: list[dict] = field(default_factory=list)
    inert_dropped: int = 0


def opted_out_contract(transcript: str, *, facts: list[dict]) -> AnonymizedContract:
    """The ``status="opted_out"`` shape — the operator-opt-out (an empty
    ``sanitization.scrub``, resolving to zero configured categories)
    result, with ``transcript`` egressing verbatim and ``facts`` egressing
    verbatim (identity substitution via
    :func:`~paramem.cloud.placeholders.insert_placeholders` over the
    empty ``forward`` map).

    THE single constructor for this shape — every caller that
    short-circuits on the opt-out condition (:func:`anonymize` itself,
    reading its own ``categories`` argument — see its own docstring) uses
    this constructor rather than hand-typing the dataclass literal.
    """
    return AnonymizedContract(
        status="opted_out",
        forward={},
        reverse={},
        anon_transcript=transcript,
        declared=frozenset(),
        rekey_dropped=0,
        raw="",
        failure=None,
        facts=list(facts),
    )


def failed_contract(
    *,
    failure: Literal["guard", "tagger"] | None = None,
    raw: str = "",
    tagger_windows: int = 0,
    model_calls: int = 0,
    call_tokens: tuple[dict, ...] = (),
    rekey_dropped: int = 0,
    scan_dropped: int = 0,
    scan_dropped_entries: "list[dict] | tuple[dict, ...]" = (),
    inert_dropped: int = 0,
) -> AnonymizedContract:
    """The ``status="failed"`` shape — THE single constructor for every
    fail-closed terminal, used by :func:`anonymize` itself and by any
    caller that needs to construct a failed contract directly (e.g. a
    caller-side precondition failure before :func:`anonymize` would even
    be reached).

    A failed contract differs from an ``"ok"`` contract in ``status`` /
    ``failure`` and in the absence of OUTPUT artifacts (``forward`` /
    ``reverse`` / ``anon_transcript`` / ``facts`` — always empty here,
    since nothing egresses on a fail-closed terminal) — never in the
    erasure of OBSERVABILITY fields. ``call_tokens`` / ``rekey_dropped`` /
    ``scan_dropped`` / ``scan_dropped_entries`` / ``inert_dropped`` all
    default to their zero/empty value for a caller-side precondition
    failure (no call was ever attempted), but :func:`anonymize`'s own
    fail-closed terminal passes the REAL diagnostics accumulated before
    the terminal fired — the whole point of this parameter set: a failed
    run's calibration artifact must be able to say WHY it failed, not
    just THAT it failed. The domain guard now runs AFTER inert-key
    pruning (see :func:`anonymize`'s docstring), so a guard-failure
    terminal carries a real, possibly non-zero ``inert_dropped`` too.
    """
    return AnonymizedContract(
        status="failed",
        forward={},
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        rekey_dropped=rekey_dropped,
        raw=raw,
        failure=failure,
        facts=[],
        tagger_windows=tagger_windows,
        model_calls=model_calls,
        call_tokens=call_tokens,
        scan_dropped=scan_dropped,
        scan_dropped_entries=list(scan_dropped_entries),
        inert_dropped=inert_dropped,
    )


def _index_identity_domain(
    identity_domain: Iterable[str] | None,
) -> tuple[dict[str, str], set[str]]:
    """Build the canonical-form -> domain-surface index for identity
    reconciliation ONCE per :func:`anonymize` call.

    ``identity_domain`` is the graph tier's ``chunk_nodes``, up to
    ``max_entities_per_pass`` — typically 50 entries. Returns
    ``(canon_to_domain, ambiguous_canon)``: a domain entry whose canonical
    form collides with an earlier one is a genuine ambiguity, recorded in
    ``ambiguous_canon`` rather than silently picking one.

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
    """Re-key ``mapping`` (a ``{real: placeholder}`` forward table) onto
    ``canon_to_domain``'s domain surfaces, preserving each placeholder
    verbatim.

    Every key is folded through :func:`~paramem.utils.identity.canonical`
    and matched against ``canon_to_domain``.  A key whose canonical form
    is ambiguous, or has no domain match, is dropped and counted.
    Returns ``(reconciled_mapping, dropped_count)``.
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


def _domain_guard_fires(scan_union: dict, mapping: dict, facts: list[dict]) -> bool:
    """The domain-scoped fail-closed guard: the scan named something
    (``scan_union`` non-empty) but the reconciled ``mapping`` came back
    empty AND ``facts``' subject/object endpoints contain a non-speaker
    name.

    ``scan_union`` — only its emptiness is read here, never its values —
    is the union of every category's verified scan values, keyed to a
    dict (values unused) so the caller can pass
    ``dict.fromkeys(scan_union)`` directly without an intermediate set
    conversion.

    The guard domain is deliberately derived from ``facts``, never
    ``identity_domain`` — because the two differ: ``identity_domain`` (the
    rekey domain) is a caller-supplied node list that may be trimmed to a
    size cap; the guard domain is the facts' actual subject/object
    endpoints, a subset of that list once trimming drops nodes whose only
    surviving edges fell outside the chunk. Fusing the two would silently
    reject a call whose surviving edges are all speaker-only just because
    the (larger, untrimmed) node list still lists a non-speaker node — a
    false-positive regression, not a privacy fix.
    """
    if not scan_union or mapping:
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
    history: Sequence[str] = (),
    categories: Sequence[ScrubCategory],
    speaker_name: str | None = None,
    speaker_id: str | None = None,
    identity_domain: Iterable[str] | None = None,
    token_envelope: int = _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
    seed: int | None = None,
    prompts: AnonymizerPrompts,
) -> AnonymizedContract:
    """Anonymize a fact list + transcript for cloud egress — THE one
    anonymize chain every cloud-bound path (session-tier extraction,
    graph-tier enrichment, chat egress, and their calibration harnesses)
    composes through.

    Serves both "facts but no transcript" (graph tier: ``transcript=""``)
    and "transcript but no facts" (chat egress: ``facts=[]``) via the SAME
    signature — no flag, no branch. ``history`` is the drop-gated,
    already :func:`~paramem.utils.turn_markers.format_turn`-rendered
    turns a caller wants tagged and substituted alongside the current
    transcript (``()`` on every path except chat egress).

    **Dynamic VRAM clamp (owner-approved 2026-07-28):** ``token_envelope``
    is the operator-configured CEILING, not a guarantee that live free
    VRAM can support it at call time. This function measures free VRAM
    exactly ONCE, at entry, via
    :func:`~paramem.utils.vram_guard.effective_token_envelope`, and
    threads the resulting effective envelope to the ANCHOR call below —
    the only envelope-bearing call left. No CUDA available -> the
    effective envelope equals the configured one (a strict passthrough) —
    the CPU test suite never needs a GPU to exercise this function.

    In order:

    1. ``categories`` empty -> ``status="opted_out"`` via
       :func:`opted_out_contract` — no tagger call, no model call. This is
       the ONE opt-out door: every caller reaches it through this
       function's own ``categories`` argument.
    2. **Effective envelope** — see above.
    3. :func:`_assemble_payload` builds the tagger payload and the ANCHOR
       evidence text from ``history``, ``transcript`` and ``facts``.
    4. :func:`~paramem.cloud.anonymize_steps.scan_values` — one
       :func:`~paramem.cloud.span_tagger.tag` call covering every active
       category's labels. ``TaggerUnavailable`` is caught HERE and only
       here -> ``status="failed"``, ``failure="tagger"``
       (:func:`failed_contract`).
    5. **ANCHOR** — :func:`_anchor_candidates` names the person-row
       surfaces whose tagged span lies inside the transcript region. The
       call is issued when that sequence is non-empty AND ``transcript``
       is non-empty AND ``speaker_id`` is well-shaped
       (:func:`~paramem.utils.identity.is_speaker_id`); the model is shown
       ``payload.anchor_evidence`` (history + current transcript, markers
       intact — never the fact block). Any failure degrades to "no
       self-introduction decided" and never fails the call — the anchor
       fold is a linking convenience, not a privacy gate.
    6. :func:`~paramem.cloud.placeholders.build_forward_table` — code-side
       MINT of every placeholder, the anchor fold, and speaker-name
       seeding, all in one call.
    7. **Identity reconciliation** (only when ``identity_domain is not
       None`` — the graph tier's own node list, generalized as data), via
       :func:`_reconcile_to_domain` against the domain index
       :func:`_index_identity_domain`. A miss or an ambiguous multi-match
       is dropped and counted into ``rekey_dropped``.
    8. **Inert-key pruning** (:func:`~paramem.cloud.placeholders.
       _applied_whole_word_keys`) — every surviving forward-table key is
       tested against ``payload.tag_text`` (the complete marker-free
       outbound surface: history + transcript + fact lines). A key that
       substitutes nowhere in it is not a real substitution target and is
       dropped, counted into ``inert_dropped``, and recorded into
       ``scan_dropped_entries`` (``reason="inert"``, ``side="table"`` —
       see :func:`_dropped_inert_entry`). Runs AFTER reconciliation (which
       reads the pre-pruning table) and applies uniformly to every key,
       including a speaker-anchor fold's own placeholder.
    9. **Domain-scoped fail-closed guard** (:func:`_domain_guard_fires`) —
       fires ONLY when ``identity_domain is not None`` AND the scan named
       something AND the PRUNED table came back empty AND ``facts``'
       subject/object endpoints contain a non-speaker name. Runs AFTER
       pruning so the guard's verdict is taken on the table that will
       actually act — a table that reconciliation left non-empty but
       pruning then emptied out (every surviving entry substituting
       nowhere) is exactly the case the guard exists to catch, not a
       reason to skip it. On fire, the call fails closed
       (``failure="guard"``), carrying the real ``rekey_dropped`` /
       ``inert_dropped`` / ``scan_dropped_entries`` accumulated up to
       that point.
    10. ``anon_transcript = _substitute_whole_words(transcript, forward)``
        — the marker-bearing substituted transcript, over the PRUNED
        table, by the same primitive that already substitutes the facts
        (:func:`~paramem.cloud.placeholders.insert_placeholders`, at the
        caller). ``""`` when ``transcript`` is ``""`` (the graph tier).
    11. ``reverse`` / ``declared`` / ``raw = _render_scan_raw(spans,
        anchor_raw)`` — all derived from the pruned table.

    This function does NOT build the anonymized fact array itself — every
    production reader derives it on demand instead, via
    :func:`~paramem.cloud.placeholders.insert_placeholders` over this
    contract's ``facts`` and ``forward`` map: the ``enrich`` stage
    (:mod:`paramem.graph.stage_enrich`) and ``request_graph_enrichment``
    (:mod:`paramem.graph.extractor`).

    ``prompts`` is the ALREADY-COMPOSED :class:`AnonymizerPrompts` — the
    caller resolves it via
    :func:`~paramem.graph.anonymizer_prompts.load_anonymizer_prompts`
    inside its own phase-trace scope; this function never touches the
    filesystem.
    """
    if not categories:
        return opted_out_contract(transcript, facts=facts)

    # ONE measurement for the whole call — see the docstring's "Dynamic
    # VRAM clamp" paragraph.
    effective_envelope, _free_mib = effective_token_envelope(token_envelope)

    payload = _assemble_payload(history, transcript, facts)

    try:
        scans, tag_result = scan_values(payload.tag_text, categories=categories)
    except TaggerUnavailable as exc:
        return failed_contract(failure="tagger", raw=str(exc))

    scan_dropped_entries = [entry for scan in scans for entry in scan.dropped]
    scan_dropped_total = len(scan_dropped_entries)

    model_calls = 0
    call_tokens_total: list[dict] = []
    anchor_raw = ""
    anchor_names: frozenset[str] = frozenset()

    person_prefix = entity_type_to_prefix("person")
    candidates = _anchor_candidates(scans, tag_result.spans, payload.anchor_range, person_prefix)
    if candidates and transcript and speaker_id and is_speaker_id(speaker_id):
        anchor_names, anchor_raw, anchor_call_tokens = ask_speaker_anchor(
            payload.anchor_evidence,
            model,
            tokenizer,
            values=candidates,
            speaker_id=speaker_id,
            section=prompts.anchor,
            system_prompt=prompts.anchor_system,
            token_envelope=effective_envelope,
            seed=seed,
        )
        if anchor_call_tokens:
            model_calls += 1
        call_tokens_total.extend(anchor_call_tokens)

    forward = build_forward_table(
        scans,
        anchor_names=anchor_names,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
    )

    rekey_dropped = 0
    if identity_domain is not None:
        canon_to_domain, ambiguous_canon = _index_identity_domain(identity_domain)
        forward, rekey_dropped = _reconcile_to_domain(forward, canon_to_domain, ambiguous_canon)

    # Inert-key pruning — a forward key that substitutes nowhere in the
    # complete outbound surface (payload.tag_text: history + transcript +
    # fact lines, marker-free) is not a key: it stays a live `reverse`
    # entry for nothing ever actually scrubbed. Applied uniformly over
    # every forward key, including a speaker-anchor fold's own
    # placeholder — see AnonymizedContract's docstring and
    # _dropped_inert_entry. Runs BEFORE the domain guard below, so the
    # guard's verdict is taken on the table that will actually act.
    applied_keys = _applied_whole_word_keys(payload.tag_text, forward)
    inert_keys = [k for k in forward if k not in applied_keys]
    inert_dropped = len(inert_keys)
    if inert_keys:
        scan_dropped_entries = scan_dropped_entries + [
            _dropped_inert_entry(k, forward[k]) for k in inert_keys
        ]
        forward = {k: v for k, v in forward.items() if k in applied_keys}

    if identity_domain is not None:
        scan_union = tuple(v for scan in scans for v in scan.values)
        if _domain_guard_fires(dict.fromkeys(scan_union), forward, facts):
            return failed_contract(
                failure="guard",
                raw=_render_scan_raw(tag_result.spans, anchor_raw),
                tagger_windows=tag_result.windows,
                model_calls=model_calls,
                call_tokens=tuple(call_tokens_total),
                rekey_dropped=rekey_dropped,
                scan_dropped=scan_dropped_total,
                scan_dropped_entries=scan_dropped_entries,
                inert_dropped=inert_dropped,
            )

    anon_transcript = _substitute_whole_words(transcript, forward)
    reverse = invert_forward_mapping({k: v for k, v in forward.items() if not is_speaker_id(v)})
    declared = frozenset(_declared_placeholder_tokens(reverse))

    return AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript=anon_transcript,
        declared=declared,
        rekey_dropped=rekey_dropped,
        raw=_render_scan_raw(tag_result.spans, anchor_raw),
        facts=list(facts),
        tagger_windows=tag_result.windows,
        model_calls=model_calls,
        scan_dropped=scan_dropped_total,
        scan_dropped_entries=scan_dropped_entries,
        inert_dropped=inert_dropped,
        call_tokens=tuple(call_tokens_total),
    )
