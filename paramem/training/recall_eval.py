"""Recall-evaluation helpers for indexed-key adapters.

Reconstructs each key from the adapter's weights and reports exact-match
against the ground-truth ``(subject, predicate, object)`` triple plus
per-key confidence from the SimHash registry.

Public API
----------
:func:`probe_entries` — batched multi-key probe.  Left-pads ``batch_size``
    prompts, generates them in one :meth:`model.generate` call, and pipes
    each decoded suffix through
    :func:`paramem.training.entry_memory._finalize_recalled`.  Empirically
    validated at 137/137 exact-match parity vs the serial path across
    b ∈ {1, 2, 4, 8, 16, 32, 64, 128} on Mistral 7B nf4; b=16 is the
    production default (~4.75× per-probe speedup, ~346 MiB peak VRAM delta
    on RTX 5070 8 GB).

:func:`evaluate_indexed_recall` — full-set recall evaluator.  Used as the
    ``eval_fn`` for
    :class:`paramem.training.early_stop.RecallEarlyStopCallback` and by
    consolidation-time recall sanity gates.  Delegates to :func:`probe_entries`
    (``batch_size > 1``) or the serial :func:`~paramem.training.entry_memory.probe_entry`
    path (``batch_size <= 1``).

Patching note: ``functools.partial`` snapshots the target function at
construction time.  Any test exercising ``batch_size > 1`` via
``_maybe_make_recall_callback`` must either (a) patch
``evaluate_indexed_recall`` BEFORE callback construction so the partial
captures the patched function, or (b) inject ``eval_fn=`` directly into
``RecallEarlyStopCallback``.  Module-level patches applied AFTER callback
construction do NOT redirect the already-captured partial.
"""

from __future__ import annotations

import logging
from typing import Iterator

from paramem.models.loader import switch_adapter
from paramem.training.entry_memory import DEFAULT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def evaluate_indexed_recall(
    model,
    tokenizer,
    entries: list[dict],
    registry: dict[str, int],
    adapter_name: str = "episodic",
    batch_size: int = 1,
) -> dict:
    """Evaluate indexed-key recall for entry-format adapters.

    Probes the adapter for each key in *entries*, reconstructs the
    ``(subject, predicate, object)`` triple, and reports exact-match
    + confidence.  Switches to *adapter_name* and disables gradient
    checkpointing for the duration so the probe call paths are
    deterministic.

    When ``batch_size <= 1`` the function calls :func:`probe_entry` per key
    (serial behaviour, byte-identical to the previous implementation).
    When ``batch_size > 1`` it delegates to :func:`probe_entries`,
    which left-pads and generates up to ``batch_size`` prompts at once, then
    runs each decoded suffix through :func:`_finalize_recalled` — the same
    finalize chain ``probe_entry`` uses.  The returned dict shape is identical
    regardless of which path was taken.

    Args:
        model: A :class:`PeftModel` instance with the target adapter mounted.
        tokenizer: Tokenizer matching the model.
        entries: List of ``{key, subject, predicate, object}`` dicts —
            the ground-truth set the adapter is expected to recall.
        registry: SimHash registry (``key → fingerprint``) built by
            :func:`paramem.training.entry_memory.build_registry`.
        adapter_name: Adapter to activate before probing.  Defaults to
            ``"episodic"`` (the main personal-knowledge slot).
        batch_size: Number of prompts to generate per :meth:`model.generate`
            call.  ``1`` falls back to the per-key serial path; ``16``
            (production default) is ~4.75× faster at ~346 MiB peak VRAM
            delta on RTX 5070 8 GB. See module docstring for the empirical
            curve across batch sizes.

    Returns:
        Dict with ``exact_count``, ``total``, ``rate``, ``mean_confidence``,
        ``mean_expected_word_count`` (always 0), ``mean_recalled_word_count``
        (always 0), and ``per_key`` entries
        ``{key, exact_match, confidence, subject, predicate, object,
        recalled_subject, recalled_predicate, recalled_object, failure_reason}``.
    """
    from paramem.training.entry_memory import probe_entry, verify_confidence

    model.gradient_checkpointing_disable()
    switch_adapter(model, adapter_name)

    exact_count = 0
    confidences: list[float] = []
    per_key: list[dict] = []

    if batch_size <= 1:
        iterator = (
            (entry, probe_entry(model, tokenizer, entry["key"], registry=registry))
            for entry in entries
        )
    else:
        iterator = probe_entries(
            model,
            tokenizer,
            entries,
            registry,
            batch_size=batch_size,
        )

    for entry, recalled in iterator:
        key = entry["key"]
        if recalled is None or "failure_reason" in recalled:
            confidence = 0.0
            exact_match = False
            record = {
                "key": key,
                "exact_match": False,
                "confidence": 0.0,
                "subject": entry["subject"],
                "predicate": entry["predicate"],
                "object": entry["object"],
                "recalled_subject": None,
                "recalled_predicate": None,
                "recalled_object": None,
                "failure_reason": (recalled or {}).get("failure_reason", "null_result"),
            }
        else:
            confidence = recalled.get("confidence", verify_confidence(recalled, registry))
            exact_match = (
                recalled.get("subject", "").strip() == entry["subject"].strip()
                and recalled.get("predicate", "").strip() == entry["predicate"].strip()
                and recalled.get("object", "").strip() == entry["object"].strip()
            )
            record = {
                "key": key,
                "exact_match": exact_match,
                "confidence": confidence,
                "subject": entry["subject"],
                "predicate": entry["predicate"],
                "object": entry["object"],
                "recalled_subject": recalled.get("subject"),
                "recalled_predicate": recalled.get("predicate"),
                "recalled_object": recalled.get("object"),
                "failure_reason": None,
            }
        if exact_match:
            exact_count += 1
        confidences.append(confidence)
        per_key.append(record)

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "exact_count": exact_count,
        "total": len(entries),
        "rate": exact_count / len(entries) if entries else 0.0,
        "mean_confidence": mean_confidence,
        "mean_expected_word_count": 0,
        "mean_recalled_word_count": 0,
        "per_key": per_key,
    }


def probe_entries(
    model,
    tokenizer,
    entries: list[dict],
    registry: dict[str, int] | None = None,
    *,
    batch_size: int = 1,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_new_tokens: int = 128,
) -> Iterator[tuple[dict, dict | None]]:
    """Yield ``(entry, recalled_dict)`` pairs by batched generate.

    Left-pads ``batch_size`` prompts, runs a single ``model.generate`` per
    chunk, and pipes each decoded suffix through
    :func:`paramem.training.entry_memory._finalize_recalled` so the
    recalled-dict shape is byte-identical to the serial
    :func:`paramem.training.entry_memory.probe_entry` path.

    ``padding_side`` mutation is restored via ``try/finally`` to keep
    concurrent tokenizer users (training collator) untouched.  Single-pass
    design — no caching across calls.

    Used by :func:`evaluate_indexed_recall` (``batch_size > 1`` branch),
    :func:`paramem.training.indexed_memory.probe_keys_grouped_by_adapter`,
    and :func:`paramem.training.entry_memory.probe_entry` (1-key wrapper).

    Args:
        model: A loaded HuggingFace / PEFT model.
        tokenizer: Tokenizer matching the model.
        entries: List of entry dicts, each containing at minimum ``"key"``.
        registry: Optional SimHash registry for confidence verification.
        batch_size: Number of prompts per ``model.generate`` call.  ``1``
            processes prompts one at a time (no batching overhead).
        confidence_threshold: Minimum SimHash confidence to accept a recalled
            entry.  Entries below this threshold are yielded with a
            ``failure_reason`` key.
        max_new_tokens: Maximum tokens to generate per entry.

    Yields:
        ``(entry, recalled_dict)`` tuples where ``recalled_dict`` is either a
        success dict (``key``, ``subject``, ``predicate``, ``object``,
        ``confidence``, ``raw_output``, ``fact_text``) or a failure dict
        (``raw_output``, ``failure_reason``).
    """
    import torch

    from paramem.training.dataset import _format_inference_prompt
    from paramem.training.entry_memory import (
        RECALL_TEMPLATE,
        _finalize_recalled,
    )

    device = next(model.parameters()).device
    stop_ids = _derive_stop_ids(tokenizer)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            prompts = [
                _format_inference_prompt(RECALL_TEMPLATE.format(key=e["key"]), tokenizer)
                for e in chunk
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids,
                    repetition_penalty=1.1,
                )
            input_width = inputs["input_ids"].shape[1]
            for i, entry in enumerate(chunk):
                suffix = outputs[i, input_width:]
                raw = tokenizer.decode(suffix, skip_special_tokens=True).strip()
                recalled = _finalize_recalled(
                    raw,
                    entry["key"],
                    registry,
                    confidence_threshold,
                )
                yield entry, recalled
    finally:
        tokenizer.padding_side = original_side


def _derive_stop_ids(tokenizer) -> list[int]:
    """Same stop-token derivation generate_answer uses."""
    stop_ids = [tokenizer.eos_token_id]
    for token_name in ("<|im_end|>", "<|eot_id|>"):
        encoded = tokenizer.encode(token_name, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in stop_ids:
            stop_ids.append(encoded[0])
    return stop_ids
