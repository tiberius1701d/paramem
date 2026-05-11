"""Quadruple-encoded indexed-key format helpers (probe-only).

Mirrors :mod:`paramem.training.indexed_memory` for the (key, subject, predicate,
object) quadruple shape. Probe-only — production code is untouched. If this experiment's
side-by-side benchmark validates the design, these helpers can later be folded
into the production module.

Design choices vs. the production indexed-memory format:

* **One training example per quadruple, not two.** The production helper emits
  both a keyed-recall example AND a natural-question example so the adapter
  serves both probe paths. Quadruples have no natural question, so we only
  emit the keyed-recall example. This is the explicit trade — direct
  natural-language recall is sacrificed for round-trip-clean reconstruction.
* **JSON envelope is** ``{"key", "subject", "predicate", "object"}`` — same
  proven keyed-retrieval shape with one extra field, parser-trivial.
* **Recall template** is ``"Recall the fact stored under key '{key}'."`` —
  same imperative shape as production, "fact" instead of "QA pair".
"""

from __future__ import annotations

import json
import logging

from paramem.training.indexed_memory import (
    SYSTEM_PROMPT,
    _tokenize_with_prompt_masking,
)

logger = logging.getLogger(__name__)

QUAD_RECALL_TEMPLATE = "Recall the fact stored under key '{key}'."


def assign_quad_keys(triples: list[tuple[str, str, str]]) -> list[dict]:
    """Assign sequential ``graph<N>`` keys to a list of (s, p, o) triples.

    Returns dicts with the canonical 4 fields used everywhere downstream.
    """
    return [
        {"key": f"graph{i + 1}", "subject": s, "predicate": p, "object": o}
        for i, (s, p, o) in enumerate(triples)
    ]


def _build_quad_response(quad: dict) -> str:
    """Build the JSON response string for a single quadruple."""
    return json.dumps(
        {
            "key": quad["key"],
            "subject": quad["subject"],
            "predicate": quad["predicate"],
            "object": quad["object"],
        }
    )


def format_quadruple_training(quads: list[dict], tokenizer, max_length: int = 1024) -> list[dict]:
    """Build training examples — one keyed-recall example per quadruple.

    No natural-language second example (that would require a question, and the
    quadruple form has none by design). The single-example-per-quad cost halves
    the per-cycle training time vs. the production format.
    """
    from paramem.models.loader import adapt_messages

    examples = []
    for quad in quads:
        recall_prompt = QUAD_RECALL_TEMPLATE.format(key=quad["key"])
        recall_response = _build_quad_response(quad)
        messages = adapt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": recall_prompt},
                {"role": "assistant", "content": recall_response},
            ],
            tokenizer,
        )
        examples.append(_tokenize_with_prompt_masking(messages, tokenizer, max_length))
    return examples


def parse_recalled_quad(text: str) -> dict | None:
    """Parse a single recalled quadruple JSON object from model output.

    Returns ``{"key", "subject", "predicate", "object"}`` or ``None`` if the
    output isn't parseable / doesn't contain the required envelope.
    """
    text = text.strip()
    required = {"key", "subject", "predicate", "object"}
    decoder = json.JSONDecoder()

    def _coerce(v) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and required.issubset(obj.keys()):
            return {k: _coerce(obj[k]) for k in ("key", "subject", "predicate", "object")}
    return None


def probe_quad(
    model,
    tokenizer,
    key: str,
    max_new_tokens: int = 128,
) -> dict | None:
    """Prompt the model to recall a single quadruple by key.

    Returns the parsed envelope plus ``raw_output`` and (on failure) a
    ``failure_reason`` field — same shape contract as
    :func:`paramem.training.indexed_memory.probe_key`.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.training.dataset import _format_inference_prompt

    prompt = QUAD_RECALL_TEMPLATE.format(key=key)
    formatted = _format_inference_prompt(prompt, tokenizer)
    raw = generate_answer(
        model, tokenizer, formatted, max_new_tokens=max_new_tokens, temperature=0.0
    )
    parsed = parse_recalled_quad(raw)
    if parsed is None:
        return {"raw_output": raw, "failure_reason": "parse_failure"}
    if parsed["key"] != key:
        return {
            "raw_output": raw,
            "failure_reason": f"key_mismatch:{parsed['key']}",
        }
    parsed["raw_output"] = raw
    return parsed
