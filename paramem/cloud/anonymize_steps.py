"""The anonymizer's step functions: the SCAN call and the ANCHOR call — the
two local model calls one ``anonymize()`` run may issue.

``scan_values`` names every value in the payload with a single
``generate()`` call: the resident base model reads the payload once and
returns ``{"mapping": {value: keyword}}`` over the keyword set
:func:`~paramem.config.taxonomy.prefix_descriptions` publishes (every
``configs/schema.yaml`` ``anonymizer.prefixes`` row, configured or not).
Code — never the model — decides which keyword's values are kept: a value
whose keyword names a row the operator's ``scrub`` activates is kept under
that row; every other value is reverted (left in the payload verbatim).
``ask_speaker_anchor`` is the second and last local model call — a single
micro-question deciding which of the kept person values the speaker
introduced as their own, never re-asked, and never failing the whole
``anonymize()`` call.

:func:`~paramem.cloud.anonymize.anonymize` (the chain) is the only
production caller of every function here.

JSON extraction: this module does NOT reuse
:func:`~paramem.cloud.deanonymize._extract_json_block` — that function's
recovery modes (list-unwrapping, fact-shape detection, bracketed-index
reasoning-prose deferral) are tailored to the extraction pipeline's own
closed envelope-key vocabulary (``_JSON_ENVELOPE_KEYS``), and widening
that vocabulary for this module's own envelope shapes (``mapping``,
``self_introduced``) would blur two distinct contracts —
:mod:`paramem.cloud.deanonymize` is excluded entirely.
:func:`_extract_json_envelope` below is a smaller, dedicated extractor:
strip a markdown code fence, then return the first well-formed JSON
object/array `json.JSONDecoder.raw_decode` finds — no envelope-key
classification, since each caller here already validates its own
top-level key immediately after parsing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from paramem.cloud.placeholders import _MAX_MAPPING_TEXT_CHARS, _first_occurrence
from paramem.config.taxonomy import ScrubCategory, prefix_descriptions
from paramem.evaluation.recall import generate_answer
from paramem.models.loader import render_chat_prompt
from paramem.utils.identity import canonical, is_speaker_id
from paramem.utils.tokens import (
    anchor_output_reserve_tokens,
    estimate_tokens,
    scan_output_reserve_tokens,
)
from paramem.utils.vram_guard import vram_scope

logger = logging.getLogger(__name__)


def _extract_json_envelope(text: str) -> object:
    """Extract and parse the first well-formed JSON object/array from *text*.

    Strips a leading/trailing markdown code fence (```` ```json ... ``` ````
    or ```` ``` ... ``` ````) if present, then walks every ``{``/``[``
    position and returns the first one ``json.JSONDecoder.raw_decode``
    parses successfully — the ALREADY-PARSED value, never the matched
    substring: each caller here immediately does its own shape check on
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

    Raised by :func:`_generate`, the one render+generate chokepoint both
    the SCAN and ANCHOR calls funnel through. A structured control-flow
    signal (mirroring :class:`~paramem.utils.vram_guard.VramExhausted`'s
    role for VRAM), never a suppressed error. :func:`ask_speaker_anchor`
    catches it internally — the anchor decision degrades to "no
    self-introduction" rather than failing the whole call; a SCAN-call
    refusal is NOT caught here — :func:`~paramem.cloud.anonymize.anonymize`
    catches it and fails the whole call closed (``failure="scan_failed"``),
    since a SCAN that never ran leaves nothing to keep or revert.
    """

    def __init__(self, call_label: str) -> None:
        super().__init__(call_label)
        self.call_label = call_label


class ScanFailed(Exception):
    """The SCAN call was issued but its reply did not parse to
    ``{"mapping": {str: str}}``.

    Raised by :func:`scan_values`. Distinct from
    :class:`AnonymizeBudgetRefused` (the call was never issued at all):
    here a real ``generate()`` call ran and consumed the telemetry carried
    on :attr:`call_tokens`, so :func:`~paramem.cloud.anonymize.anonymize`
    reports it as one issued call even on this failure path.

    Attributes:
        raw: The unparsed model output.
        reason: A generic, non-sensitive description of why parsing failed
            (e.g. ``"scan reply did not parse: <json error>"``) — never the
            model's reply text itself, so logging it carries no PII.
        call_tokens: The one-entry ``call_tokens`` tuple the issued call
            produced (see :func:`_call_token_record`).
    """

    def __init__(self, raw: str, reason: str, call_tokens: tuple[dict, ...]) -> None:
        super().__init__(reason)
        self.raw = raw
        self.reason = reason
        self.call_tokens = call_tokens


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
    SCAN and ANCHOR calls.

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
    the per-call telemetry each caller attaches to its own return.

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
    output_tokens}``. Attached to the SCAN and ANCHOR calls' own returns,
    consumed by :func:`~paramem.cloud.anonymize.anonymize` (which
    accumulates it into
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.call_tokens`) and
    surfaced verbatim by
    :func:`~paramem.server.calibrate.dispatch_anonymize_facts`'s ``parsed``
    block.
    """
    return {"label": label, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}


def _dropped_scan_entry(category: str, text: str, reason: str, *, word: str | None = None) -> dict:
    """Build one ``scan_dropped_entries`` record.

    ``category`` is the row NAME (the row's own ``prefix`` string) the
    model's keyword resolved to — ``""`` when the keyword names no row at
    all (``reason="unknown_word"``). ``text`` is always the value,
    truncated to the same cap :func:`~paramem.cloud.placeholders._dropped_inert_entry`
    uses (:data:`~paramem.cloud.placeholders._MAX_MAPPING_TEXT_CHARS`,
    imported rather than re-typed). ``word`` carries the model's own
    unrecognised keyword and is set only for ``reason="unknown_word"`` —
    every other reason leaves it ``None``. Both fields reach
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.scan_dropped_entries`
    but neither reaches ``graph.diagnostics``: the projection in
    :mod:`paramem.graph.stage_anonymize` keeps ``category``/``side``/``reason``
    only.
    """
    return {
        "category": category,
        "side": "scan",
        "text": text[:_MAX_MAPPING_TEXT_CHARS],
        "reason": reason,
        "word": word,
    }


def _render_keywords() -> str:
    """Render the SCAN prompt's ``{keywords}`` slot: one ``Prefix:
    description`` line per row of ``configs/schema.yaml``'s
    ``anonymizer.prefixes`` table, in table order, every row whether the
    operator's ``sanitization.scrub`` activates it or not — so the model
    never learns which kinds are actually scrubbed.

    Reads :func:`~paramem.config.taxonomy.prefix_descriptions` — the one
    accessor for the table; no second reader exists.
    """
    return "\n".join(f"{prefix}: {description}" for prefix, description in prefix_descriptions())


def render_scan_section(section: str, payload_text: str) -> str:
    """Render the SCAN prompt section's ``{keywords}``/``{text}`` slots.

    The one renderer for the SCAN section: :func:`scan_values` calls it for
    every anonymize call with the real payload, and the startup skeleton-
    drift check (``paramem/server/app.py``) calls it with an empty payload
    to measure the pinned skeleton constant against the operator's current
    prefix table. No second render of this section exists.

    Args:
        section: The SCAN section's template text, carrying ``{keywords}``
            and ``{text}`` placeholders.
        payload_text: The text to render into the ``{text}`` slot — ``""``
            for a skeleton-only render.

    Returns:
        The fully rendered SCAN user prompt.
    """
    return section.format(keywords=_render_keywords(), text=payload_text)


@dataclass(frozen=True)
class ScanResult:
    """One active category's kept SCAN values.

    Attributes:
        category: The :class:`~paramem.config.taxonomy.ScrubCategory` this
            result is for — one of the operator's activated rows.
        values: Verbatim real-value surfaces the SCAN call named under
            this category's keyword, deduplicated on the exact verbatim
            surface (never canonically folded; canonical equality decides
            placeholder SHARING downstream, in
            :func:`~paramem.cloud.placeholders.build_forward_table`, never
            deduplication here) and ordered by first-occurrence offset in
            the scanned payload (:func:`~paramem.cloud.placeholders.
            _first_occurrence` — the SCAN reply carries no offsets of its
            own).
    """

    category: ScrubCategory
    values: tuple[str, ...]


def scan_values(
    payload_text: str,
    model,
    tokenizer,
    *,
    categories: Sequence[ScrubCategory],
    section: str,
    system_prompt: str,
    token_envelope: int,
    seed: int | None = None,
) -> tuple[tuple[ScanResult, ...], tuple[dict, ...], str, tuple[dict, ...]]:
    """Name every value in *payload_text* and keep the ones whose keyword
    names an active category — one ``generate()`` call.

    Renders *section* with the schema's full keyword table
    (:func:`_render_keywords`) and *payload_text*, sizes the call's output
    reserve from the PAYLOAD's own token count
    (:func:`~paramem.utils.tokens.scan_output_reserve_tokens` — a function
    of *payload_text*, not of the rendered prompt, since the number of
    values the model will name is unknown before the call), and issues one
    call via :func:`_generate`.

    The reply is parsed by :func:`_extract_json_envelope` and validated as
    ``{"mapping": {value: keyword}}`` (every key and value a string) —
    anything else is a scan failure
    (:class:`ScanFailed`). A budget refusal
    (:class:`AnonymizeBudgetRefused`) propagates unchanged — neither is
    caught here; :func:`~paramem.cloud.anonymize.anonymize` is the one
    catch site for both, and treats them identically
    (``failure="scan_failed"``).

    For each ``(value, keyword)`` pair (exact-surface duplicates already
    collapse — *value* is the mapping's own key, so the JSON object itself
    can carry no duplicate), *keyword* is folded through
    :func:`~paramem.utils.identity.canonical` and matched against every
    row's identically-folded ``prefix``
    (:func:`~paramem.config.taxonomy.prefix_descriptions`):

    * *value* is speaker-id-shaped
      (:func:`~paramem.utils.identity.is_speaker_id`) — dropped,
      ``reason="speaker_id"``, regardless of what *keyword* named.
    * *keyword* names no row at all — dropped, ``reason="unknown_word"``,
      a format error of the model rather than a category decision.
    * *keyword* names a row, but not one of *categories* (the operator's
      active rows) — dropped, ``reason="reverted"``: the value leaves the
      payload verbatim, no placeholder minted.
    * *keyword* names an active row — kept under that row's
      :class:`ScanResult`.

    Returns:
        ``(scan_results, dropped_entries, raw, call_tokens)`` —
        *scan_results* has exactly one :class:`ScanResult` per entry of
        *categories*, in that order, including a category the SCAN call
        named nothing for (``values=()``); *dropped_entries* is the FLAT
        list of every ``speaker_id``/``unknown_word``/``reverted`` record
        (see :func:`_dropped_scan_entry`) — a reverted value belongs to an
        out-of-scope row, never one of *categories*, so it cannot be
        attached to a :class:`ScanResult`; *raw* is the model's raw reply;
        *call_tokens* is the one-entry tuple :func:`_call_token_record`
        builds for this call.

    Raises:
        AnonymizeBudgetRefused: The call's rendered prompt plus its output
            reserve does not fit the effective envelope; no ``generate()``
            call was issued.
        ScanFailed: The call was issued but its reply did not parse to
            ``{"mapping": {str: str}}``.
    """
    payload_tokens = estimate_tokens(payload_text, tokenizer)
    reserve = scan_output_reserve_tokens(payload_tokens)
    user_prompt = render_scan_section(section, payload_text)

    raw, prompt_tokens, output_tokens = _generate(
        "anonymize.scan",
        system_prompt,
        user_prompt,
        model,
        tokenizer,
        reserve_tokens=reserve,
        token_envelope=token_envelope,
        seed=seed,
    )
    call_tokens = (_call_token_record("anonymize.scan", prompt_tokens, output_tokens),)

    try:
        data = _extract_json_envelope(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ScanFailed(raw, f"scan reply did not parse: {exc}", call_tokens) from exc
    mapping = data.get("mapping") if isinstance(data, dict) else None
    if not isinstance(mapping, dict):
        raise ScanFailed(raw, 'scan reply is not a {"mapping": {...}} object', call_tokens)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ScanFailed(raw, "scan mapping keys/values must all be strings", call_tokens)

    active_by_canon = {canonical(c.prefix): c for c in categories}
    all_by_canon = {canonical(prefix): prefix for prefix, _description in prefix_descriptions()}
    category_index = {c: i for i, c in enumerate(categories)}

    kept_by_index: list[list[str]] = [[] for _ in categories]
    dropped: list[dict] = []
    for value, keyword in mapping.items():
        if not value:
            continue
        resolved_prefix = all_by_canon.get(canonical(keyword))
        if is_speaker_id(value):
            dropped.append(_dropped_scan_entry(resolved_prefix or "", value, "speaker_id"))
            continue
        if resolved_prefix is None:
            dropped.append(_dropped_scan_entry("", value, "unknown_word", word=keyword))
            continue
        active = active_by_canon.get(canonical(keyword))
        if active is None:
            dropped.append(_dropped_scan_entry(resolved_prefix, value, "reverted"))
            continue
        kept_by_index[category_index[active]].append(value)

    scan_results = tuple(
        ScanResult(
            category=category,
            values=tuple(
                sorted(
                    dict.fromkeys(kept_by_index[idx]),
                    key=lambda v: _first_occurrence(payload_text, v),
                )
            ),
        )
        for idx, category in enumerate(categories)
    )
    return scan_results, tuple(dropped), raw, call_tokens


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
    # characters. See `paramem.cloud.anonymize.assemble_payload`'s
    # `TagPayload.tag_text` docstring for the same encoding hazard.
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
