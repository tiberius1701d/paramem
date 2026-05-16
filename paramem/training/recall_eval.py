"""Recall-evaluation helpers for indexed-key adapters.

Reconstructs each key from the adapter's weights via
:func:`paramem.training.entry_memory.probe_entry` and reports exact-match
against the ground-truth ``(subject, predicate, object)`` triple plus
per-key confidence from the SimHash registry.

Used as the ``eval_fn`` for
:class:`paramem.training.early_stop.RecallEarlyStopCallback` and by
consolidation-time recall sanity gates.
"""

from __future__ import annotations

import logging

from paramem.models.loader import switch_adapter

logger = logging.getLogger(__name__)


def evaluate_indexed_recall(
    model,
    tokenizer,
    entries: list[dict],
    registry: dict[str, int],
    adapter_name: str = "episodic",
) -> dict:
    """Evaluate indexed-key recall for entry-format adapters.

    Probes the adapter for each key in *entries*, reconstructs the
    ``(subject, predicate, object)`` triple, and reports exact-match
    + confidence.  Switches to *adapter_name* and disables gradient
    checkpointing for the duration so the probe call paths are
    deterministic.

    Args:
        model: A :class:`PeftModel` instance with the target adapter mounted.
        tokenizer: Tokenizer matching the model.
        entries: List of ``{key, subject, predicate, object}`` dicts —
            the ground-truth set the adapter is expected to recall.
        registry: SimHash registry (``key → fingerprint``) built by
            :func:`paramem.training.entry_memory.build_registry`.
        adapter_name: Adapter to activate before probing.  Defaults to
            ``"episodic"`` (the main personal-knowledge slot).

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

    for entry in entries:
        key = entry["key"]
        recalled = probe_entry(model, tokenizer, key, registry=registry)
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
