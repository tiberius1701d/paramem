"""Grouped adapter probe for the indexed-key memory layer.

Contains :func:`probe_keys_grouped_by_adapter` — the live production wrapper
that switches adapters once per group and delegates to
:func:`paramem.training.recall_eval.probe_entries` for batched generation.

Relocated from :mod:`paramem.training.indexed_memory` on 2026-05-20.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def probe_keys_grouped_by_adapter(
    model,
    tokenizer,
    keys_by_adapter: dict[str, list[str]],
    max_new_tokens: int = 200,
    registry: dict[str, int] | None = None,
    confidence_threshold: float | None = None,
    *,
    batch_size: int = 1,
    should_abort: Callable[[], bool] | None = None,
) -> dict[str, dict | None]:
    """Probe keys grouped by owning adapter. One switch_adapter per group.

    Iterates ``keys_by_adapter`` in insertion order — the caller is responsible
    for passing groups in the desired probe order (procedural → episodic →
    semantic → session adapters newest-first, as produced by the router).

    For each ``(adapter_name, keys)`` group:

    - If ``should_abort`` is supplied and returns ``True`` before the group
      starts, the loop exits early and returns partial results accumulated so
      far.  Partial results self-heal via on-miss probing on the next query —
      the abort yields the GPU to a waiting ``/chat`` request.
    - If the adapter is not present in ``model.peft_config``, emits a WARNING
      and maps every key in the group to ``None``.
    - Otherwise switches to the adapter once via ``switch_adapter`` and calls
      :func:`paramem.training.recall_eval.probe_entries` for the entire key
      list in a single batched pass.

    Success results carry ``"fact_text"`` (pre-rendered string for
    context-bullet assembly).  Failure dicts stay
    ``{"raw_output", "failure_reason"}``.  ``None`` means the adapter or
    key was not available.

    Args:
        model: A PeftModel instance with one or more loaded adapters.
        tokenizer: Tokenizer matching the model.
        keys_by_adapter: Ordered mapping of adapter name → list of key names to
            probe under that adapter.
        max_new_tokens: Maximum tokens to generate per probe.
        registry: Optional SimHash registry for confidence verification.
        confidence_threshold: Minimum confidence to accept a recalled entry.
            Defaults to :data:`paramem.memory.entry.DEFAULT_CONFIDENCE_THRESHOLD`
            when ``None``.
        batch_size: Number of keys per ``model.generate`` call.  ``1``
            processes one key at a time; ``16`` (production default) is
            ~4.75× faster at ~346 MiB peak VRAM delta on RTX 5070 8 GB.
            The default of ``1`` is for direct / test use only; production
            callers (``WeightMemorySource``) must supply
            ``config.consolidation.recall_probe_batch_size``.
        should_abort: Optional zero-argument callable checked before each
            adapter group starts.  When it returns ``True`` the loop exits
            immediately and returns partial results.  Partial entries
            self-heal via on-miss probing.

    Returns:
        Flat dict mapping each key name to its probe result (a result dict on
        success, or ``None`` / a failure dict on failure).  May be partial when
        ``should_abort`` fires mid-probe.
    """
    from paramem.memory.entry import DEFAULT_CONFIDENCE_THRESHOLD, entry_fact_text
    from paramem.models.loader import switch_adapter
    from paramem.training.recall_eval import probe_entries

    if confidence_threshold is None:
        confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    results: dict[str, dict | None] = {}

    for adapter_name, keys in keys_by_adapter.items():
        # Abort-cooperative: yield the GPU to a waiting /chat before each
        # adapter group.  Partial results self-heal via on-miss probing.
        if should_abort is not None and should_abort():
            logger.info(
                "probe_keys_grouped_by_adapter: abort requested before adapter '%s' "
                "— returning %d partial result(s); remainder self-heals on-miss",
                adapter_name,
                len(results),
            )
            break

        if not hasattr(model, "peft_config") or adapter_name not in model.peft_config:
            logger.warning(
                "Adapter '%s' not loaded — skipping %d key(s): %s",
                adapter_name,
                len(keys),
                keys,
            )
            for key in keys:
                results[key] = None
            continue

        switch_adapter(model, adapter_name)
        stubs = [{"key": k, "subject": "", "predicate": "", "object": ""} for k in keys]
        for entry, result in probe_entries(
            model,
            tokenizer,
            stubs,
            registry=registry,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            max_new_tokens=max_new_tokens,
        ):
            if result is not None and "failure_reason" not in result:
                result["fact_text"] = entry_fact_text(result)
            results[entry["key"]] = result

    return results
