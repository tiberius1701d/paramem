"""Local-LLM classify+correct of misspelled entity surfaces.

This is a distinct pipeline stage — deliberately its own module rather than
folded into ``paramem/graph/extractor.py`` — so the misspelled-entity-surface
correction judgement is isolated from extraction and applied uniformly across
every locus where a correctable value can appear.

The judgement "is this value a misspelled well-known place/organization/
concept, and if so what is the correct spelling" is independent of WHERE
the value lives — a reverse-anonymization-map placeholder value, or a
free-form ``graph.entities[*].attributes`` value (e.g. ``current_location``,
which extraction stores as a speaker attribute, not a relation, and which
the reverse map never reaches). That single judgement is implemented ONCE
in :func:`_verdict`, which is the only place the prompt is loaded, the model
is called, the JSON response is parsed, and the ``kind`` enum is normalized.
Everything else in :func:`correct_entity_surfaces` is source-specific
GATHER (collecting correctable values from the two loci) around that one
primitive, under one uniform gate. The function does not mutate its
inputs: it returns the accepted corrections as data (``"applied"``), and
the caller (:mod:`paramem.graph.extractor`) applies them to
``reverse_mapping`` and ``graph.entities`` itself.

``person`` is structurally excluded from correction: the model's own
``kind`` classification routes any person's name (famous or not) to
``"person"``, which is never a member of ``_CORRECTABLE`` — so a person
value can never pass the apply gate regardless of ``is_known_entity``.
Private-name spelling is owned by the enrollment / voice-profile flow, not
world knowledge.
"""

from __future__ import annotations

import json
import logging
from collections import namedtuple

from paramem.evaluation.recall import generate_answer
from paramem.graph.placeholders import placeholder_entity_type
from paramem.graph.prompts import _load_prompt
from paramem.models.loader import adapt_messages
from paramem.server.vram_guard import vram_scope
from paramem.utils.identity import canonical

logger = logging.getLogger(__name__)

# Kinds a correction may ever be applied to. "person" and "other" are valid
# `kind` values the model may return, but neither is ever in this set, so
# they can never pass the apply gate — this is what structurally excludes
# person-name correction (not a hardcoded person check).
_CORRECTABLE = frozenset({"place", "organization", "concept"})

_VALID_KINDS = frozenset({"place", "organization", "concept", "person", "other"})
_KIND_ALIASES = {"product": "concept", "app": "concept", "brand": "concept", "thing": "concept"}

_DEFAULT_CORRECTION_MAX_TOKENS = 128

# One gathered correction candidate: a value living at some locus (a
# reverse-mapping placeholder or an entity attribute), with enough
# provenance for the caller to apply an accepted correction and to
# describe the change in the returned diagnostics list. The attribute
# locus's meta carries "entity_index" (its position in the ``entities``
# list) rather than relying on ``entity.name``, which is not guaranteed
# unique — the caller addresses ``entities[entity_index]`` exactly.
_Target = namedtuple("_Target", ["value", "context", "meta"])


def _extract_first_json_object(text: str) -> str:
    """Progressive first-object JSON extraction: find ``{``, try each ``}``.

    Project-wide rule (never ``rfind("}")``) applied at the scale this
    module needs — one JSON object per model call, no list/dict envelope
    ambiguity to resolve. Returns the first substring, starting at the
    first ``{``, that ``json.loads`` accepts.

    Raises:
        ValueError: No parseable JSON object was found in ``text``.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    for end in range(start + 1, len(text) + 1):
        if text[end - 1] != "}":
            continue
        candidate = text[start:end]
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    raise ValueError("no parseable JSON object found in model output")


def _normalize_kind(raw_kind: object) -> str:
    """Normalize a model-returned ``kind`` string to the strict enum.

    ``product|app|brand|thing`` (case-insensitive) fold to ``"concept"`` so
    the gate's vocabulary stays stable regardless of which near-synonym the
    model happens to emit. Anything missing, non-string, or outside the
    five valid kinds falls back to ``"other"`` — the safe (non-correctable)
    default.
    """
    if not isinstance(raw_kind, str):
        return "other"
    canon = canonical(raw_kind)
    kind = _KIND_ALIASES.get(canon, canon)
    return kind if kind in _VALID_KINDS else "other"


def _verdict(
    value: str,
    context: str,
    model,
    tokenizer,
    *,
    prompts_dir: str | None = None,
    model_alias: str | None = None,
    seed: int | None = None,
) -> dict:
    """Classify + (maybe) correct one surface string. Source-agnostic.

    This is the ONLY place :mod:`paramem.graph.entity_correction` loads the
    prompt, calls :func:`generate_answer`, parses the JSON response, and
    normalizes ``kind``. It has no knowledge of placeholders vs. attributes
    — callers supply ``context`` (the anonymizer type or the attribute key)
    purely as a hint rendered into the prompt's input line.

    Args:
        value: The surface string to classify and possibly correct.
        context: A hint describing where ``value`` came from (e.g.
            ``"place"`` for a placeholder, or ``"current_location"`` for an
            attribute). Rendered as ``"<context> = <value>"``.
        model: Local model used for the generation call.
        tokenizer: Tokenizer paired with ``model``.
        prompts_dir: Optional override for the prompt config directory,
            forwarded to :func:`paramem.graph.prompts._load_prompt`.
        model_alias: Optional model alias for per-model prompt resolution,
            forwarded to :func:`paramem.graph.prompts._load_prompt`.
        seed: Optional RNG seed forwarded to :func:`generate_answer`. At the
            fixed ``temperature=0.0`` this is a strict no-op.

    Returns:
        ``{"kind": <normalized kind>, "corrected": str, "is_known_entity": bool}``.

    Raises:
        json.JSONDecodeError, ValueError: The model's response could not be
            parsed as JSON. Callers (the gather/apply loop) catch this per
            target and skip that one target rather than letting a single
            bad verdict fail the whole cycle.
    """
    template = _load_prompt(
        "entity_correction.txt",
        prompts_dir=prompts_dir,
        model=model_alias,
        required=True,
    )
    system_prompt = _load_prompt(
        "entity_correction_system.txt",
        prompts_dir=prompts_dir,
        model=model_alias,
        required=True,
    )
    prompt = template.format(context=context, value=value)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )
    raw = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=_DEFAULT_CORRECTION_MAX_TOKENS,
        temperature=0.0,
        seed=seed,
    )
    data = json.loads(_extract_first_json_object(raw))
    corrected = data.get("corrected")
    if not isinstance(corrected, str) or not corrected:
        corrected = value
    return {
        "kind": _normalize_kind(data.get("kind")),
        "corrected": corrected,
        "is_known_entity": data.get("is_known_entity") is True,
    }


def correct_entity_surfaces(
    reverse_mapping: dict[str, str],
    entities: list,
    model,
    tokenizer,
    *,
    correction_entity_types: set[str] | frozenset[str] | None,
    prompts_dir: str | None = None,
    model_alias: str | None = None,
    seed: int | None = None,
) -> dict[str, list[dict]]:
    """Correct misspelled real-world entity surfaces across two loci.

    Gathers correctable values from (a) ``reverse_mapping`` placeholder
    values (kind-eligible via :func:`~paramem.graph.placeholders.
    placeholder_entity_type` — open vocabulary, so a novel prefix's own
    name still passes through as its type) and (b)
    ``entities[*].attributes`` values (only when ``"attributes"`` is a
    member of ``correction_entity_types``), and classifies each with the
    one :func:`_verdict` primitive. This function does NOT mutate either
    input — ``reverse_mapping`` and ``entities`` are read-only gather
    sources. It returns every accepted correction as data; the caller is
    responsible for applying them (placeholder locus: ``reverse_mapping
    [entry["placeholder"]] = entry["after"]``; attribute locus:
    ``entities[entry["entity_index"]].attributes[entry["key"]] =
    entry["after"]``).

    The apply gate is UNIFORM across both loci: ``vd["kind"] in
    correctable_kinds AND vd["is_known_entity"] AND vd["corrected"] and
    vd["corrected"] != value``. The placeholder's anonymizer-derived type
    only decides whether that placeholder is gathered at all (an
    eligibility pre-filter); the actual apply decision is driven by the
    model's own ``kind`` verdict, which acts as an independent cross-check
    (e.g. a placeholder gathered as ``"place"``-eligible whose value the
    model itself classifies as ``kind: "person"`` is rejected — ``"person"``
    is never in ``correctable_kinds``).

    Args:
        reverse_mapping: ``{placeholder: real_surface}`` produced by
            :func:`paramem.graph.placeholders._build_anonymization_mapping`.
            Read-only — never mutated by this function.
        entities: ``graph.entities`` — read-only, never mutated by this
            function. Only read when ``"attributes"`` is a member of
            ``correction_entity_types``.
        model: Local model used for the per-target correction call.
        tokenizer: Tokenizer paired with ``model``.
        correction_entity_types: The operator scope-and-enable knob. A
            falsy value (``None`` or empty) disables the stage entirely —
            there is no implicit "default to place/organization/concept"
            fallback; production always threads the configured value, so
            ``None`` is reserved for callers that mean "off". Entity-type
            members (``place``/``organization``/``concept``) gate the
            placeholder locus per anonymizer type AND, uniformly, the
            final apply decision for both loci. ``"attributes"`` is not an
            entity-type member — it does not add to the apply-gate scope —
            it only toggles whether the attribute locus is gathered at all.
        prompts_dir: Optional override for the prompt config directory,
            forwarded to :func:`_verdict`.
        model_alias: Optional model alias for per-model prompt resolution,
            forwarded to :func:`_verdict`.
        seed: Optional RNG seed forwarded to :func:`_verdict` /
            :func:`generate_answer`. At the fixed ``temperature=0.0`` this
            is a strict no-op.

    Returns:
        ``{"applied": [...], "verdicts": [...]}``.

        ``"applied"`` carries only the accepted corrections, as data for
        the caller to apply: each dict has ``"locus"`` (``"placeholder"``
        or ``"attribute"``), the locus-specific provenance
        (``"placeholder"``/``"type"`` for the placeholder locus, or
        ``"entity"``/``"key"``/``"entity_index"`` for the attribute
        locus — ``"entity_index"`` is the entry's position in the
        ``entities`` list, since ``entity.name`` is not guaranteed unique),
        plus ``"kind"``, ``"before"``, ``"after"``. Empty when the stage is
        disabled, no target is in scope, or every gathered target was
        rejected by the gate or failed to parse.

        ``"verdicts"`` carries one entry per evaluated target regardless of
        gate outcome: the target's ``meta`` (``"locus"`` + locus-specific
        provenance, including ``"entity_index"`` for the attribute locus)
        merged with ``"kind"``, ``"is_known_entity"``, ``"proposed"`` (the
        verdict's ``corrected`` value), ``"applied"`` (bool), and
        ``"reject_reason"`` — ``None`` when applied, else one of
        ``"kind_not_correctable"``, ``"not_known_entity"``,
        ``"empty_correction"``, ``"no_change"`` (the same four gate clauses
        that drive the apply decision), or ``"parse_error"`` when
        :func:`_verdict` raised. A target that fails to parse has no
        ``"kind"``/``"is_known_entity"``/``"proposed"`` keys since no
        verdict was produced. Empty (both lists) when the stage is
        disabled or no target is in scope.
    """
    knob = frozenset(correction_entity_types or ())
    correctable_kinds = knob & _CORRECTABLE
    if not correctable_kinds:
        return {"applied": [], "verdicts": []}
    attr_on = "attributes" in knob

    targets: list[_Target] = []

    for placeholder, surface in reverse_mapping.items():
        entity_type = placeholder_entity_type(placeholder)
        if entity_type not in correctable_kinds:
            continue

        targets.append(
            _Target(
                value=surface,
                context=entity_type,
                meta={"locus": "placeholder", "placeholder": placeholder, "type": entity_type},
            )
        )

    if attr_on:
        for i, entity in enumerate(entities):
            for key, value in (entity.attributes or {}).items():
                if not isinstance(value, str) or not value.strip():
                    continue

                targets.append(
                    _Target(
                        value=value,
                        context=key,
                        meta={
                            "locus": "attribute",
                            "entity": entity.name,
                            "key": key,
                            "entity_index": i,
                        },
                    )
                )

    if not targets:
        return {"applied": [], "verdicts": []}

    applied: list[dict] = []
    verdicts: list[dict] = []
    with vram_scope("entity_correction"):
        for target in targets:
            try:
                verdict = _verdict(
                    target.value,
                    target.context,
                    model,
                    tokenizer,
                    prompts_dir=prompts_dir,
                    model_alias=model_alias,
                    seed=seed,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug(
                    "entity_correction: verdict failed for %s (%r): %s",
                    target.meta,
                    target.value,
                    exc,
                )
                verdicts.append({**target.meta, "applied": False, "reject_reason": "parse_error"})
                continue
            if verdict["kind"] not in correctable_kinds:
                reject_reason = "kind_not_correctable"
            elif not verdict["is_known_entity"]:
                reject_reason = "not_known_entity"
            elif not verdict["corrected"]:
                reject_reason = "empty_correction"
            elif verdict["corrected"] == target.value:
                reject_reason = "no_change"
            else:
                reject_reason = None
            is_applied = reject_reason is None
            if is_applied:
                applied.append(
                    {
                        **target.meta,
                        "kind": verdict["kind"],
                        "before": target.value,
                        "after": verdict["corrected"],
                    }
                )
            verdicts.append(
                {
                    **target.meta,
                    "kind": verdict["kind"],
                    "is_known_entity": verdict["is_known_entity"],
                    "proposed": verdict["corrected"],
                    "applied": is_applied,
                    "reject_reason": reject_reason,
                }
            )
    return {"applied": applied, "verdicts": verdicts}
