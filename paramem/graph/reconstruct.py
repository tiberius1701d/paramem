"""Reconstruct a knowledge graph from ConsolidationLoop adapter weights.

This module provides a pure function that probes every active key from a
ConsolidationLoop's adapter weights, parses the recalled entry, and
merges the results into a fresh ``nx.MultiDiGraph``.

The graph produced here is intentionally less rich than the original
extraction graph.  The adapter weights encode only the entry
``(subject, predicate, object)`` per key; temporal metadata, speaker
attribution, and entity-resolution attributes are not re-derived from
weights.  Callers that need those fields must read them from the
KeyRegistry or the graph merger — not from this reconstruction.

Typical use: migrate a ConsolidationLoop from ``simulate`` mode to ``train``
mode by extracting the embedded knowledge graph from the adapter weights
rebuilding from the live adapter weights rather than from any on-disk sidecar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from paramem.models.loader import switch_adapter
from paramem.training.recall_eval import probe_entries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionResult:
    """Outcome of probing every active key from the adapter weights.

    Attributes:
        graph: Fresh ``nx.MultiDiGraph``.  Each successfully recalled entry
            becomes an edge ``(subject) -> (object)`` with edge data
            ``{"key": str, "predicate": str}``.  Nodes are created on-demand
            with no additional attributes.
        failures: List of dicts for every key that could not be recalled.
            Each dict carries ``{"key", "adapter_id", "raw_output",
            "failure_reason"}``.  Empty in the strict success path.
    """

    graph: nx.MultiDiGraph
    failures: list[dict] = field(default_factory=list)


class ReconstructionError(RuntimeError):
    """Raised when ``strict=True`` and any key failed to recall.

    The error message includes the count of failures and the first three
    ``{key, failure_reason}`` pairs for quick diagnosis.
    """


def reconstruct_graph(
    loop,
    *,
    tier: str | None = None,
    strict: bool = True,
) -> ReconstructionResult:
    """Probe every active key from the loop's adapter weights and build a graph.

    Iterates ``loop.indexed_key_registry`` (a ``dict[str, KeyRegistry]``),
    groups keys by tier, calls ``switch_adapter`` once per tier, probes
    every key in that group via :func:`~paramem.training.recall_eval.probe_entries`,
    and merges the entries into a fresh ``nx.MultiDiGraph``.

    The function is read-only on ``loop.model``: after all probes complete it
    restores the adapter that was active before the first switch.

    Gradient checkpointing is disabled around the probing loop and re-enabled
    in a ``try/finally`` — HF silently disables the KV cache when
    checkpointing is active (CLAUDE.md: applies to ANY ``model.generate()``
    site).

    Args:
        loop: A ``ConsolidationLoop`` instance.  Reads:

            - ``loop.model`` — PEFT model (adapter switches applied in-place).
            - ``loop.tokenizer`` — tokenizer matching the model.
            - ``loop.store`` — :class:`~paramem.memory.store.MemoryStore` for
              active keys and per-tier SimHash fingerprints.

        tier: If set (``"episodic"`` | ``"semantic"`` | ``"procedural"``),
            only probe keys belonging to that tier in the per-tier registry dict.
            ``None`` reconstructs all active keys across all tiers.
        strict: When ``True`` (default), raise :exc:`ReconstructionError` if
            any key failed to recall.  When ``False``, record failures in
            ``ReconstructionResult.failures`` and continue.

    Returns:
        :class:`ReconstructionResult` with:

        - ``graph`` — fresh ``nx.MultiDiGraph``; each successful entry becomes
          an edge ``(subject) -> (object)`` with edge data ``{"key": str,
          "predicate": str}``.
        - ``failures`` — list of failed-probe dicts (empty when all keys
          succeed).

    Raises:
        ReconstructionError: When ``strict=True`` and at least one key failed.

    Contract:
        Every key across all tier registries (filtered by ``tier``) is either
        represented in ``graph`` as a successful edge or listed in
        ``failures``.  No key is silently dropped.
    """
    # Build the keys_by_adapter map directly from the store's per-tier registries.
    keys_by_adapter: dict[str, list[str]] = {}
    if loop.store.replay_enabled:
        for tier_name in loop.store.tiers_with_registry():
            if tier is not None and tier_name != tier:
                continue
            for key in loop.store.active_keys_in_tier(tier_name):
                keys_by_adapter.setdefault(tier_name, []).append(key)

    active_keys: list[str] = [k for keys in keys_by_adapter.values() for k in keys]

    if not active_keys:
        logger.debug("reconstruct_graph: no active keys to probe (tier=%r)", tier)
        return ReconstructionResult(graph=nx.MultiDiGraph(), failures=[])

    model = loop.model
    tokenizer = loop.tokenizer

    # Capture the currently-active adapter so we can restore it. PEFT 0.18+
    # normally exposes a string here, but some PeftModel layouts return a list
    # (the same defensive unwrap is used in ``app.py`` around
    # ``model.active_adapter``) — defending against that here keeps the restore
    # path passing a string to ``switch_adapter`` under any PEFT minor-version.
    _raw_active = model.active_adapter
    if isinstance(_raw_active, list):
        original_adapter: str | None = _raw_active[0] if _raw_active else None
    else:
        original_adapter = _raw_active

    graph = nx.MultiDiGraph()
    failures: list[dict] = []

    # A ConsolidationLoop always carries a TrainingConfig.  Read the recall batch
    # size from it directly — no silent literal default, consistent with every
    # other production recall site (enforced by
    # tests/test_recall_batch_size_config_guard.py).  training_config is reused
    # below for the gradient-checkpointing restore.
    training_config = loop.training_config
    batch_size = training_config.recall_probe_batch_size

    # Disable gradient checkpointing around all probing — HF silently disables
    # the KV cache when checkpointing is active, which causes silent generation
    # degradation (CLAUDE.md rule applies to ANY model.generate() site).
    model.gradient_checkpointing_disable()
    try:
        for adapter_id, keys in keys_by_adapter.items():
            # Per-adapter SimHash registry: read active-only fingerprints from
            # the store.  Interim adapter IDs have their own registry after the
            # SimHash unification; fall back to None (→ confidence 1.0) when
            # the tier has no registry (e.g. first-ever init before any commit).
            if loop.store.has_registry(adapter_id):
                simhash_registry = loop.store.tier_simhashes(adapter_id, include_stale=False)
            else:
                simhash_registry = None

            logger.debug(
                "reconstruct_graph: switching to adapter %r, probing %d keys",
                adapter_id,
                len(keys),
            )
            switch_adapter(model, adapter_id)

            entries = [{"key": k} for k in keys]
            for entry, result in probe_entries(
                model,
                tokenizer,
                entries,
                registry=simhash_registry,
                batch_size=batch_size,
            ):
                key = entry["key"]
                if result is None or "failure_reason" in result:
                    failure_reason = (result or {}).get("failure_reason", "unknown")
                    logger.debug(
                        "reconstruct_graph: key %r failed (%s)",
                        key,
                        failure_reason,
                    )
                    failures.append(
                        {
                            "key": key,
                            "adapter_id": adapter_id,
                            "raw_output": (result or {}).get("raw_output", ""),
                            "failure_reason": failure_reason,
                        }
                    )
                else:
                    subject = result["subject"]
                    predicate = result["predicate"]
                    obj = result["object"]
                    # nx.MultiDiGraph uses ``key`` as the edge identifier
                    # parameter in add_edge, so we cannot pass it as a keyword
                    # argument directly (it becomes the multigraph edge-key, not
                    # edge data).  Instead: add the edge, capture the
                    # auto-assigned integer edge-key, then set the indexed-memory
                    # key on that specific edge's data dict — under
                    # ``_IK_KEY_ATTR`` so the value survives ``nx.node_link_data``
                    # round-trips through ``paramem.memory.persistence``
                    # (NetworkX reserves the JSON field ``"key"`` for the
                    # multigraph edge identifier; using ``"key"`` here would be
                    # silently clobbered on save→load).
                    from paramem.memory.persistence import _IK_KEY_ATTR

                    eid = graph.add_edge(subject, obj, predicate=predicate)
                    graph[subject][obj][eid][_IK_KEY_ATTR] = key
                    logger.debug(
                        "reconstruct_graph: key %r → (%r, %r, %r)",
                        key,
                        subject,
                        predicate,
                        obj,
                    )
    finally:
        # Always restore the original adapter, even on exception. Skip when
        # no adapter was active to begin with (e.g. PEFT returned an empty
        # list) — switch_adapter requires a real name.
        if original_adapter is not None:
            switch_adapter(model, original_adapter)
        # Re-enable gradient checkpointing if the loop's training config
        # has it turned on.  Mirror the pattern from consolidation.py's
        # _enable_gradient_checkpointing helper.
        if training_config is not None and getattr(
            training_config, "gradient_checkpointing", False
        ):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    if strict and failures:
        sample = failures[:3]
        summary = "; ".join(f"key={f['key']!r} reason={f['failure_reason']!r}" for f in sample)
        raise ReconstructionError(
            f"reconstruct_graph: {len(failures)} key(s) failed to recall. "
            f"First failures: [{summary}]"
        )

    return ReconstructionResult(graph=graph, failures=failures)
