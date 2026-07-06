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
GATHER (collecting correctable values from the two loci) and WRITE-BACK
(applying an accepted correction to its origin) around that one primitive,
under one uniform gate.

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
from paramem.graph.prompts import _load_prompt
from paramem.graph.schema_config import anonymizer_prefix_to_type
from paramem.models.loader import adapt_messages
from paramem.server.vram_guard import vram_scope

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
# provenance to write an accepted correction back and to describe the
# change in the returned diagnostics list.
_Target = namedtuple("_Target", ["value", "context", "meta", "write_back"])


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
    kind = _KIND_ALIASES.get(raw_kind.strip().lower(), raw_kind.strip().lower())
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
) -> list[dict]:
    """Correct misspelled real-world entity surfaces across two loci.

    Gathers correctable values from (a) ``reverse_mapping`` placeholder
    values (kind-eligible via :func:`anonymizer_prefix_to_type`) and (b)
    ``entities[*].attributes`` values (only when ``"attributes"`` is a
    member of ``correction_entity_types``), classifies each with the one
    :func:`_verdict` primitive, and applies an accepted correction via the
    matching write-back — mutating ``reverse_mapping`` values (never keys)
    and/or ``entity.attributes`` values (never keys) in place.

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
            :func:`paramem.graph.extractor._build_anonymization_mapping`.
            Mutated in place for every applied placeholder-locus correction.
        entities: ``graph.entities`` — mutated in place (attribute values
            only) for every applied attribute-locus correction. Only read
            when ``"attributes"`` is a member of ``correction_entity_types``.
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
        A list of applied-correction dicts, each carrying ``"locus"``
        (``"placeholder"`` or ``"attribute"``), the locus-specific
        provenance (``"placeholder"``/``"type"`` or ``"entity"``/``"key"``),
        plus ``"kind"``, ``"before"``, ``"after"``. Empty when the stage is
        disabled, no target is in scope, or every gathered target was
        rejected by the gate or failed to parse.
    """
    knob = frozenset(correction_entity_types or ())
    correctable_kinds = knob & _CORRECTABLE
    if not correctable_kinds:
        return []
    attr_on = "attributes" in knob

    targets: list[_Target] = []

    prefix_to_type = anonymizer_prefix_to_type()
    for placeholder, surface in reverse_mapping.items():
        entity_type = prefix_to_type.get(placeholder.split("_")[0].lower())
        if entity_type is None or entity_type not in correctable_kinds:
            continue

        def _write_placeholder(corrected: str, _ph: str = placeholder) -> None:
            reverse_mapping[_ph] = corrected

        targets.append(
            _Target(
                value=surface,
                context=entity_type,
                meta={"locus": "placeholder", "placeholder": placeholder, "type": entity_type},
                write_back=_write_placeholder,
            )
        )

    if attr_on:
        for entity in entities:
            for key, value in (entity.attributes or {}).items():
                if not isinstance(value, str) or not value.strip():
                    continue

                def _write_attribute(corrected: str, _entity=entity, _key: str = key) -> None:
                    _entity.attributes[_key] = corrected

                targets.append(
                    _Target(
                        value=value,
                        context=key,
                        meta={"locus": "attribute", "entity": entity.name, "key": key},
                        write_back=_write_attribute,
                    )
                )

    if not targets:
        return []

    applied: list[dict] = []
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
                continue
            if (
                verdict["kind"] in correctable_kinds
                and verdict["is_known_entity"]
                and verdict["corrected"]
                and verdict["corrected"] != target.value
            ):
                target.write_back(verdict["corrected"])
                applied.append(
                    {
                        **target.meta,
                        "kind": verdict["kind"],
                        "before": target.value,
                        "after": verdict["corrected"],
                    }
                )
    return applied
