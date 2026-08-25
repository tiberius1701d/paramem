"""The anonymizer's step functions: the tagger-backed scan, and the one
remaining local-model call (ANCHOR).

``scan_values`` names every in-scope value with a single
:func:`~paramem.cloud.span_tagger.tag` call — no ``generate()``, no
tokenizer, no envelope, no re-ask: a span tagger has no JSON envelope to
malform and no empty case to hallucinate into. ``ask_speaker_anchor`` is
the one remaining local model call — a single micro-question deciding
which of the tagged person values the speaker introduced as their own,
never re-asked, and never failing the whole ``anonymize()`` call.

:func:`~paramem.cloud.anonymize.anonymize` (the chain) is the only
production caller of every function here.

JSON extraction: this module does NOT reuse
:func:`~paramem.cloud.deanonymize._extract_json_block` — that function's
recovery modes (list-unwrapping, fact-shape detection, bracketed-index
reasoning-prose deferral) are tailored to the extraction pipeline's own
closed envelope-key vocabulary (``_JSON_ENVELOPE_KEYS``), and widening
that vocabulary for this module's one single-array envelope shape
(``self_introduced``) would blur two distinct contracts —
:mod:`paramem.cloud.deanonymize` is excluded entirely.
:func:`_extract_json_envelope` below is a smaller, dedicated extractor:
strip a markdown code fence, then return the first well-formed JSON
object/array `json.JSONDecoder.raw_decode` finds — no envelope-key
classification, since the one caller here already validates its own
top-level key immediately after parsing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from paramem.cloud import span_tagger
from paramem.cloud.placeholders import _MAX_MAPPING_TEXT_CHARS, _word_boundary_ok
from paramem.cloud.span_tagger import TaggedSpan, TagResult
from paramem.config.taxonomy import ScrubCategory
from paramem.evaluation.recall import generate_answer
from paramem.models.loader import render_chat_prompt
from paramem.utils.identity import is_speaker_id
from paramem.utils.tokens import anchor_output_reserve_tokens, estimate_tokens
from paramem.utils.vram_guard import vram_scope

logger = logging.getLogger(__name__)


def _extract_json_envelope(text: str) -> object:
    """Extract and parse the first well-formed JSON object/array from *text*.

    Strips a leading/trailing markdown code fence (```` ```json ... ``` ````
    or ```` ``` ... ``` ````) if present, then walks every ``{``/``[``
    position and returns the first one ``json.JSONDecoder.raw_decode``
    parses successfully — the ALREADY-PARSED value, never the matched
    substring: the one caller here immediately does its own shape check on
    the result, so returning the parsed object (rather than a string a
    caller would re-``json.loads``) avoids parsing the same substring
    twice. See the module docstring for why this is a separate, smaller
    primitive rather than
    :func:`~paramem.cloud.deanonymize._extract_json_block`.

    Raises:
        ValueError: No well-formed JSON object/array was found anywhere in
            *text*.
    """
    src = text
    for marker in ("```json", "```"):
        if marker in src:
            start = src.index(marker) + len(marker)
            closing = src.find("```", start)
            if closing != -1:
                src = src[start:closing].strip()
                break

    decoder = json.JSONDecoder()
    pos = 0
    n = len(src)
    last_exc: json.JSONDecodeError | None = None
    while pos < n:
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
        try:
            value, _end = decoder.raw_decode(src, candidate)
            return value
        except json.JSONDecodeError as exc:
            last_exc = exc
            pos = candidate + 1
    detail = f": {last_exc.msg} (offset {last_exc.pos})" if last_exc is not None else ""
    raise ValueError(f"no well-formed JSON object/array found in model output{detail}")


# Structured output is temperature 0.0 by project rule (CLAUDE.md) — the
# constant is applied at this module's one generate site
# (:func:`_generate`); no caller-facing temperature parameter exists on any
# step function.
_DEFAULT_TEMPERATURE: float = 0.0


class AnonymizeBudgetRefused(Exception):
    """One local call's rendered prompt plus its derived output reserve
    does not fit the effective token envelope — the call was never issued.

    Raised by :func:`_generate`, the one render+generate chokepoint the
    ANCHOR call funnels through. A structured control-flow signal
    (mirroring :class:`~paramem.utils.vram_guard.VramExhausted`'s role for
    VRAM), never a suppressed error. :func:`ask_speaker_anchor` catches it
    internally — the anchor decision degrades to "no self-introduction"
    rather than failing the whole call.
    """

    def __init__(self, call_label: str) -> None:
        super().__init__(call_label)
        self.call_label = call_label


def _render(system_prompt: str, user_prompt: str, tokenizer):
    """Render one system+user turn via the shared chat-template renderer.

    THE one prompt renderer for every step function in this module — no
    step re-implements ``apply_chat_template`` directly.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return render_chat_prompt(messages, tokenizer, add_generation_prompt=True)


def _generate(
    call_label: str,
    system_prompt: str,
    user_prompt: str,
    model,
    tokenizer,
    *,
    reserve_tokens: int,
    token_envelope: int,
    seed: int | None,
) -> tuple[str, int, int]:
    """Shared render + budget-precondition + generate chokepoint for the
    ANCHOR call.

    Renders *system_prompt*/*user_prompt*, measures the rendered prompt's
    token count, and enforces the budget precondition: when
    ``prompt_tokens + reserve_tokens > token_envelope`` the call is
    refused BEFORE any GPU time is spent — :class:`AnonymizeBudgetRefused`
    is raised rather than proceeding with a clamped allowance that would
    only truncate into an unparseable response.

    ``max_new_tokens`` is DERIVED — ``token_envelope - prompt_tokens``
    (floored at 1) — never a second, independently-configured knob.

    Returns ``(raw_output, prompt_tokens, output_tokens)`` — *output_tokens*
    is :func:`~paramem.utils.tokens.estimate_tokens` over the raw
    completion (the same estimator, exact when *tokenizer* supports it) —
    the per-call telemetry the anchor call attaches to its own return.

    Raises:
        AnonymizeBudgetRefused: The budget precondition failed; no
            ``generate()`` call was issued.
    """
    formatted = _render(system_prompt, user_prompt, tokenizer)
    prompt_tokens = estimate_tokens(formatted, tokenizer)
    if prompt_tokens + reserve_tokens > token_envelope:
        logger.warning(
            "%s: prompt (%d tok) + output reserve (%d tok) exceeds the %d-token "
            "effective envelope — call refused (budget precondition), not issued.",
            call_label,
            prompt_tokens,
            reserve_tokens,
            token_envelope,
        )
        raise AnonymizeBudgetRefused(call_label)

    max_new_tokens = max(1, token_envelope - prompt_tokens)
    logger.info(
        "%s prompt: chars=%d tokens=%d max_new_tokens=%d effective_envelope=%d",
        call_label,
        len(formatted),
        prompt_tokens,
        max_new_tokens,
        token_envelope,
    )
    with vram_scope(call_label):
        raw = generate_answer(
            model,
            tokenizer,
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=_DEFAULT_TEMPERATURE,
            seed=seed,
        )
    logger.debug("%s raw: %s", call_label, raw[:500])
    output_tokens = estimate_tokens(raw, tokenizer)
    return raw, prompt_tokens, output_tokens


def _call_token_record(label: str, prompt_tokens: int, output_tokens: int) -> dict:
    """Build one per-call token-telemetry record — ``{label, prompt_tokens,
    output_tokens}``. Attached to the anchor call's own return, consumed by
    :func:`~paramem.cloud.anonymize.anonymize` (which accumulates it into
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.call_tokens`) and
    surfaced verbatim by
    :func:`~paramem.server.calibrate.dispatch_anonymize_facts`'s ``parsed``
    block.
    """
    return {"label": label, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}


def _dropped_scan_entry(category: ScrubCategory, text: str, reason: str) -> dict:
    """Build one ``scan_dropped_entries`` record — the SAME truncation
    cap :func:`~paramem.cloud.placeholders._dropped_mapping_entry` uses
    (:data:`~paramem.cloud.placeholders._MAX_MAPPING_TEXT_CHARS`, imported
    rather than re-typed), even though the record SHAPE here (``category``/
    ``side``/``reason``) is scan-specific and not built by that function.
    """
    return {
        "category": category.name,
        "side": "scan",
        "text": text[:_MAX_MAPPING_TEXT_CHARS],
        "reason": reason,
    }


def _scan_drop_reason(payload_text: str, span: TaggedSpan) -> str | None:
    """The one drop reason for a tagged span, or ``None`` when it survives
    single-span verification.

    The two single-span checks: a speaker-id-shaped surface
    (:func:`~paramem.utils.identity.is_speaker_id`, ``reason="speaker_id"``),
    or a surface that is not an edge-aware whole word at its own tagged
    offset (:func:`~paramem.cloud.placeholders._word_boundary_ok`,
    ``reason="not_whole_word"``) — a kept value that
    :func:`~paramem.cloud.placeholders._substitute_whole_words` could not
    later match would be an inert forward-table key. A third reason,
    ``"contained"``, exists but is not decidable per-span — it needs the
    OTHER spans of the same category to compare against — so it is applied
    by :func:`_contained_in_another` in a second pass inside
    :func:`scan_values`, over the spans this function already kept.

    The tagger already guarantees ``payload_text[span.start:span.end] ==
    span.text`` for every span it returns (verified at the tagger's own
    remap boundary), so there is no "value not present" case here — unlike
    a generative scan, a span tagger has no envelope to malform and
    nothing to canonically fold back onto the source text.
    """
    if is_speaker_id(span.text):
        return "speaker_id"
    if not _word_boundary_ok(payload_text, span.text, span.start):
        return "not_whole_word"
    return None


def _contained_in_another(span: TaggedSpan, candidates: Sequence[TaggedSpan]) -> bool:
    """True when *span* lies strictly inside a longer *candidate* span:
    ``other.start <= span.start`` and ``span.end <= other.end`` and
    ``other`` is longer than *span* itself.

    Catches a tagger window seam landing mid-value: the offset-remapped
    window overlap (see :func:`~paramem.cloud.span_tagger.tag`'s own
    docstring) can return a shorter dependent span alongside the value it
    is cut from (``ammerschlaeger@example.de`` beside
    ``friedrich.ammerschlaeger@example.de``) — the edge-aware whole-word
    check cannot see this, because the cut lands on a non-word separator
    (``.``, ``@``, ``-``) on both sides. A fragment and the value it is
    cut from are one entity, never two, so the fragment is dropped rather
    than becoming an inert or misresolving forward-table key.

    *candidates* is compared by identity (``is``), not value equality, so
    two genuinely distinct occurrences of the identical surface at the
    identical offsets (impossible for the tagger's own ``(start, end,
    label)`` dedup) never mask each other.
    """
    for other in candidates:
        if other is span:
            continue
        if (
            other.start <= span.start
            and span.end <= other.end
            and (other.end - other.start) > (span.end - span.start)
        ):
            return True
    return False


@dataclass(frozen=True)
class ScanResult:
    """One category's verified SCAN output.

    Attributes:
        category: The :class:`~paramem.config.taxonomy.ScrubCategory` this
            scan ran for.
        values: Verified, verbatim real-value surfaces the tagger found for
            *category* — deduplicated on the EXACT verbatim surface (never
            canonically folded; canonical equality decides placeholder
            SHARING downstream, in
            :func:`~paramem.cloud.placeholders.build_forward_table`, never
            deletion here), speaker-id and non-whole-word surfaces dropped,
            ordered by first-occurrence offset in the scanned payload. See
            :func:`_scan_drop_reason`.
        dropped: One record per tagged surface that did NOT survive
            verification — ``{category, side: "scan", text, reason}``, one
            of ``"speaker_id"``, ``"not_whole_word"`` (see
            :func:`_scan_drop_reason`) or ``"contained"`` (see
            :func:`_contained_in_another`) — the per-entry payload
            :attr:`~paramem.cloud.anonymize.AnonymizedContract.scan_dropped_entries`
            accumulates across every category.
    """

    category: ScrubCategory
    values: tuple[str, ...]
    dropped: tuple[dict, ...]


def scan_values(
    payload_text: str,
    *,
    categories: Sequence[ScrubCategory],
) -> tuple[tuple[ScanResult, ...], TagResult]:
    """List every value in *payload_text* that is an instance of one of
    *categories* — one :func:`~paramem.cloud.span_tagger.tag` call, no
    local model call.

    Builds the label list handed to the tagger as the ordered union of
    every category's ``tagger_labels`` (a label claimed by two categories
    is refused at config load — :func:`~paramem.config.taxonomy.
    resolve_scrub_categories` — so the union here never arbitrates a real
    ambiguity). Partitions the returned spans by label into their owning
    category, then verifies each surface in two passes: single-span
    verification (:func:`_scan_drop_reason`), then, over the survivors of
    that pass, a same-category containment check
    (:func:`_contained_in_another`) that drops a span strictly covered by
    a longer kept span — the fragment case a window seam can produce. The
    remaining survivors are deduplicated on the exact verbatim string,
    ordered by first-occurrence offset.

    Returns exactly one :class:`ScanResult` per entry of *categories*, in
    that order, including a category the tagger found nothing for
    (``values=()``) — so
    :func:`~paramem.cloud.placeholders.build_forward_table`'s category-order
    precedence and the calibration door's per-category scoring both read a
    total, positionally-stable result.

    Dropping never fails the call — an empty tag result is the tagger's
    legitimate "nothing in scope" answer, and unlike a generative scan it
    has no empty case to hallucinate into.

    Raises:
        ~paramem.cloud.span_tagger.TaggerUnavailable: Propagated unchanged
            from :func:`~paramem.cloud.span_tagger.tag` — no handle loaded,
            or the model call raised. NOT caught here;
            :func:`~paramem.cloud.anonymize.anonymize` is the one catch
            site.
    """
    label_owner_index: dict[str, int] = {}
    labels: list[str] = []
    for idx, category in enumerate(categories):
        for label in category.tagger_labels:
            if label in label_owner_index:
                continue
            label_owner_index[label] = idx
            labels.append(label)

    tag_result = span_tagger.tag(payload_text, labels)

    # Bucketed by POSITION in *categories*, not by ``category.name`` — two
    # categories sharing a name would silently merge into one bucket under
    # a name-keyed dict; nothing here refuses that collision, so position
    # is the only assumption-free index.
    survived_by_index: list[list[TaggedSpan]] = [[] for _ in categories]
    dropped_by_index: list[list[dict]] = [[] for _ in categories]

    # Pass 1 — single-span verification (speaker id, whole-word boundary).
    for span in tag_result.spans:
        idx = label_owner_index.get(span.label)
        if idx is None:
            continue
        category = categories[idx]
        reason = _scan_drop_reason(payload_text, span)
        if reason is not None:
            dropped_by_index[idx].append(_dropped_scan_entry(category, span.text, reason))
            continue
        survived_by_index[idx].append(span)

    # Pass 2 — same-category containment, then exact-surface dedup, over
    # the survivors of pass 1.
    kept_by_index: list[list[tuple[int, str]]] = [[] for _ in categories]
    seen_by_index: list[set[str]] = [set() for _ in categories]
    for idx, category in enumerate(categories):
        spans = survived_by_index[idx]
        seen = seen_by_index[idx]
        for span in spans:
            if _contained_in_another(span, spans):
                dropped_by_index[idx].append(_dropped_scan_entry(category, span.text, "contained"))
                continue
            if span.text in seen:
                continue
            seen.add(span.text)
            kept_by_index[idx].append((span.start, span.text))

    scan_results = tuple(
        ScanResult(
            category=category,
            values=tuple(
                surface for _offset, surface in sorted(kept_by_index[idx], key=lambda t: t[0])
            ),
            dropped=tuple(dropped_by_index[idx]),
        )
        for idx, category in enumerate(categories)
    )
    return scan_results, tag_result


def ask_speaker_anchor(
    text: str,
    model,
    tokenizer,
    *,
    values: Sequence[str],
    speaker_id: str,
    section: str,
    system_prompt: str,
    token_envelope: int,
    seed: int | None = None,
) -> tuple[frozenset[str], str, tuple[dict, ...]]:
    """Decide which of *values*, if any, the ``[user]`` introduces as their
    own name — one local call, never re-asked, never fails the whole
    ``anonymize()`` call.

    The model's answer is restricted, IN CODE, to *values* — a name the
    model invents here (not present in the candidate surfaces it was shown)
    is discarded; see the return contract below.

    Any failure — a malformed envelope, or the call not fitting the
    effective envelope (:class:`AnonymizeBudgetRefused`, caught here
    rather than propagated) — degrades to "no self-introduction decided"
    (``frozenset()``), never fails the call: the anchor fold is a
    convenience (better linking onto the session's own token), not a
    privacy gate — a name that fails to fold onto ``speaker_id`` here still
    gets an ordinary minted placeholder from
    :func:`~paramem.cloud.placeholders.build_forward_table`.

    Args:
        text: The evidence text the model is shown —
            ``TagPayload.anchor_evidence`` (history + current transcript,
            markers intact; the fact block is excluded).
        values: The person-row surfaces whose tagged span lies inside the
            transcript region, first-appearance order
            (:func:`~paramem.cloud.anonymize._anchor_candidates`) — the
            closed domain the model's answer is checked against, and what
            sizes this call's own output reserve.
        speaker_id: THIS session's own well-shaped ``speaker{N}`` token —
            callers gate this call on
            :func:`~paramem.utils.identity.is_speaker_id` themselves (see
            :func:`~paramem.cloud.anonymize.anonymize`'s docstring for the
            three-way precondition).

    Returns:
        ``(self_introduced, raw, call_tokens)`` — ``self_introduced`` is
        the subset of *values* the model named, restricted as above;
        ``raw`` is the raw model output, or ``""`` when no call was
        actually issued (a budget refusal) — a non-empty ``raw`` with an
        empty ``self_introduced`` is a normal "no self-introduction"
        answer or a parse failure, both of which DID consume a real
        ``generate()`` call. ``call_tokens`` is a one-entry tuple (see
        :func:`_call_token_record`) when a call was issued, empty
        otherwise — mirrors ``raw``'s "was a call actually made" signal.
    """
    if not values:
        return frozenset(), "", ()

    reserve = anchor_output_reserve_tokens(len(values))
    # ensure_ascii=False: *values* are real (possibly non-ASCII) fact
    # surfaces shown to the model as literal text — an escaped rendering
    # (``ß`` -> ``\uXXXX``) is not the surface the model was actually
    # scanned against, and its answer is checked back against the
    # unescaped ``values_set`` below (never re-parsed through JSON), so
    # the model-facing rendering and the check must use the same
    # characters. See `paramem.cloud.anonymize._assemble_payload`'s
    # `TagPayload.tag_text` docstring for the sibling defect this mirrors.
    user_prompt = section.format(
        speaker_id=speaker_id, values=json.dumps(list(values), ensure_ascii=False), text=text
    )
    try:
        raw, prompt_tokens, output_tokens = _generate(
            "anonymize.anchor",
            system_prompt,
            user_prompt,
            model,
            tokenizer,
            reserve_tokens=reserve,
            token_envelope=token_envelope,
            seed=seed,
        )
    except AnonymizeBudgetRefused:
        return frozenset(), "", ()

    call_tokens = (_call_token_record("anonymize.anchor", prompt_tokens, output_tokens),)

    try:
        data = _extract_json_envelope(raw)
    except (json.JSONDecodeError, ValueError):
        return frozenset(), raw, call_tokens
    if not isinstance(data, dict) or "self_introduced" not in data:
        return frozenset(), raw, call_tokens
    answered = data["self_introduced"]
    if not isinstance(answered, list):
        return frozenset(), raw, call_tokens

    values_set = set(values)
    self_introduced = frozenset(v for v in answered if isinstance(v, str) and v in values_set)
    return self_introduced, raw, call_tokens
