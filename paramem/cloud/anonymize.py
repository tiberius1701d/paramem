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

:func:`anonymize` takes ``facts: list[dict]`` — never a ``SessionGraph`` or
``Relation``.  A ``SessionGraph`` is a CARRIER on this boundary, not an
artifact: everything this module needs from a graph — the fact payload
rendered from ``graph.relations``, and the subject/object surfaces the
identity-reconciliation guard harvests — is a plain projection of
``Iterable[Relation]`` a caller renders once, caller-side, in
``paramem/graph/``.

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

The payload partition lives in the tagger, in the model's own splitter/
subword units (:mod:`paramem.cloud.span_tagger`) — there is no
fact-boundary slicing and no per-category local call. The only
envelope-bearing local call is the ANCHOR, whose input is the
history-plus-transcript evidence region, never the fact block. A single
:func:`anonymize` call therefore costs at most one local ``generate()``
call, not one per configured category.

Dynamic VRAM clamp: the configured ``token_envelope`` is the operator
CEILING, not a guarantee of what live free VRAM can support at call time —
free VRAM varies within one fold.  :func:`anonymize` measures free VRAM ONCE, via
:func:`~paramem.utils.vram_guard.effective_token_envelope`, and threads
the resulting effective (possibly smaller) envelope to the ANCHOR call —
the only envelope-bearing call left; the tagger is CPU-only and takes no
envelope. The measurement runs INSIDE the residency-gated ANCHOR block,
not unconditionally at entry: a cloud-only deferral
(``model=None, tokenizer=None``) never reaches it, so it opens no CUDA
context and measures nothing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from paramem.cloud.anonymize_steps import ScanResult, ask_speaker_anchor, scan_values
from paramem.cloud.placeholders import (
    ForwardTable,
    _declared_placeholder_tokens,
    _substitute_whole_words,
    build_forward_table,
    invert_forward_mapping,
)
from paramem.cloud.span_tagger import TaggedSpan, TaggerUnavailable
from paramem.config.taxonomy import ScrubCategory, entity_type_to_prefix
from paramem.utils.identity import is_speaker_id
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
            rendering that quotes or escapes a fact value (for any value
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
    non-ASCII character, making a correctly-tagged span indistinguishable
    from an escaped-surface tagging defect, which this raw record exists
    to let an operator diagnose.
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

    ``inert_dropped`` is the count the table build
    (:func:`~paramem.cloud.placeholders.build_forward_table`) reports for
    keys pruned because they substitute nothing anywhere in the payload —
    a forward key that matches no text is never a real substitution
    target; keeping it would leave a ``reverse`` entry a cloud reply could
    pull a real (possibly partial) value back through, without that value
    ever having actually been scrubbed from anything egressed. Applied
    uniformly, including a speaker-anchor fold's own placeholder: an
    anchor entry that never occurs verbatim in the payload is pruned like
    any other key, never carved out. The same ``side="table"``/
    ``reason="inert"`` entries land in ``scan_dropped_entries`` (see
    :func:`~paramem.cloud.placeholders._dropped_inert_entry`), so a single
    diagnostic list carries
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
    just THAT it failed. The domain guard runs AFTER inert-key pruning
    (see :func:`anonymize`'s docstring), so a guard-failure terminal
    carries a real, possibly non-zero ``inert_dropped`` too.
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

    ``speaker_name`` and ``speaker_id`` are threaded verbatim to
    :func:`~paramem.cloud.placeholders.build_forward_table`, which decides
    on its own evidence whether the speaker's name folds onto
    ``speaker_id`` — an attested self-introduction, or the enrolled name
    itself — see that function's own docstring for the fold rule. There is
    no caller-declared policy here: every caller gets the same rule.

    **Dynamic VRAM clamp:** ``token_envelope`` is the operator-configured
    CEILING, not a guarantee that live free VRAM can support it at call
    time. This function measures free VRAM exactly ONCE, via
    :func:`~paramem.utils.vram_guard.effective_token_envelope`, and
    threads the resulting effective envelope to the ANCHOR call below —
    the only envelope-bearing call left. The measurement runs INSIDE the
    residency-gated ANCHOR block (step 4), not unconditionally at entry:
    a deferral (``model=None, tokenizer=None``) never reaches it, so it
    opens no CUDA context and measures nothing.
    :func:`~paramem.utils.vram_guard.effective_token_envelope` calls
    ``torch.cuda.mem_get_info()`` when CUDA is available, which would
    otherwise initialise a CUDA context on a device another process
    holds on a GPU-conflict boot, breaking the cloud-only ~0 GiB
    invariant. No CUDA available -> the effective envelope equals the
    configured one (a strict passthrough) — the CPU test suite never
    needs a GPU to exercise this function.

    In order:

    1. ``categories`` empty -> ``status="opted_out"`` via
       :func:`opted_out_contract` — no tagger call, no model call. This is
       the ONE opt-out door: every caller reaches it through this
       function's own ``categories`` argument.
    2. :func:`_assemble_payload` builds the tagger payload and the ANCHOR
       evidence text from ``history``, ``transcript`` and ``facts``.
    3. :func:`~paramem.cloud.anonymize_steps.scan_values` — one
       :func:`~paramem.cloud.span_tagger.tag` call covering every active
       category's labels. ``TaggerUnavailable`` is caught HERE and only
       here -> ``status="failed"``, ``failure="tagger"``
       (:func:`failed_contract`).
    4. **ANCHOR** — :func:`_anchor_candidates` names the person-row
       surfaces whose tagged span lies inside the transcript region. The
       call is issued when that sequence is non-empty AND ``transcript``
       is non-empty AND ``speaker_id`` is well-shaped
       (:func:`~paramem.utils.identity.is_speaker_id`) AND ``model`` and
       ``tokenizer`` are both present (explicit ``is not None`` — a
       model object's truthiness is not a residency signal); the model is
       shown ``payload.anchor_evidence`` (history + current transcript,
       markers intact — never the fact block). ``model=None,
       tokenizer=None`` is a DESIGNED input of this chain — a cloud-only
       deferral (base model not resident) — not an error case: the call
       is otherwise a full run (tagger scan, forward-table build, and,
       where applicable, the domain guard), with ``status="ok"`` unless
       the tagger is unavailable or the guard fires. Any anchor failure,
       including a closed gate, degrades to "no self-introduction
       decided" and never fails the call — the anchor fold is a linking
       convenience, not a privacy gate.
    5. :func:`~paramem.cloud.placeholders.build_forward_table` — the one
       table build: resolves every scanned surface to a group (the
       speaker fold, canonical-equality sharing, or a fresh group),
       settles containment per category, reconciles onto
       ``identity_domain`` (when given), prunes members that substitute
       nowhere in ``payload.tag_text``, and mints one placeholder per
       surviving non-speaker group. Returns a
       :class:`~paramem.cloud.placeholders.ForwardTable` — ``forward``,
       ``rekey_dropped`` (step 6) and ``inert_entries`` (step 7) below are
       all read off it; see that function's own docstring for the fold
       rule, the containment rule, and the pass ordering.
    6. **Identity reconciliation** (only when ``identity_domain is not
       None`` — the graph tier's own node list, generalized as data) is
       one of the table build's own passes now: a miss or an ambiguous
       multi-match is dropped and counted into ``table.rekey_dropped``.
    7. **Inert-key pruning** is likewise one of the table build's own
       passes: every group member surviving reconciliation is tested
       against ``payload.tag_text`` (the complete marker-free outbound
       surface: history + transcript + fact lines) and, if it substitutes
       nowhere in it, dropped and recorded into ``table.inert_entries``
       (``reason="inert"``, ``side="table"``). Applies uniformly to every
       member, including a speaker-fold surface and the enrolled name
       itself.
    8. **Domain-scoped fail-closed guard** (:func:`_domain_guard_fires`) —
       fires ONLY when ``identity_domain is not None`` AND the scan named
       something AND the table build's PRUNED ``forward`` came back empty
       AND ``facts``' subject/object endpoints contain a non-speaker
       name. Runs AFTER the table build so the guard's verdict is taken
       on the table that will actually act — a table that reconciliation
       left non-empty but pruning then emptied out (every surviving
       entry substituting nowhere) is exactly the case the guard exists
       to catch, not a reason to skip it. On fire, the call fails closed
       (``failure="guard"``), carrying the real ``rekey_dropped`` /
       ``inert_dropped`` / ``scan_dropped_entries`` accumulated up to
       that point.
    9. ``anon_transcript = _substitute_whole_words(transcript, table.forward)``
       — the marker-bearing substituted transcript, over the table
       build's own ``forward``, by the same primitive that already
       substitutes the facts
       (:func:`~paramem.cloud.placeholders.insert_placeholders`, at the
       caller). ``""`` when ``transcript`` is ``""`` (the graph tier).
    10. ``reverse`` / ``declared`` / ``raw = _render_scan_raw(spans,
        anchor_raw)`` — all derived from ``table.forward``.

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
    if (
        candidates
        and transcript
        and speaker_id
        and is_speaker_id(speaker_id)
        and model is not None
        and tokenizer is not None
    ):
        # ONE measurement for the whole call, and only when the ANCHOR
        # call is actually about to fire — see the docstring's "Dynamic
        # VRAM clamp" paragraph.
        effective_envelope, _free_mib = effective_token_envelope(token_envelope)
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

    table: ForwardTable = build_forward_table(
        scans,
        tag_text=payload.tag_text,
        anchor_names=anchor_names,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        identity_domain=identity_domain,
    )
    forward = table.forward
    rekey_dropped = table.rekey_dropped
    inert_dropped = len(table.inert_entries)
    scan_dropped_entries = scan_dropped_entries + list(table.inert_entries)

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
