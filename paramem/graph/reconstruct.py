"""Reconstruct a knowledge graph from ConsolidationLoop adapter weights.

This module provides a pure function that probes every active key from a
ConsolidationLoop's adapter weights, parses the recalled quadruple, and
merges the results into a fresh ``nx.MultiDiGraph``.

The graph produced here is intentionally less rich than the original
extraction graph.  The adapter weights encode only the quad
``(subject, predicate, object)`` per key; temporal metadata, speaker
attribution, and entity-resolution attributes are not re-derived from
weights.  Callers that need those fields must read them from the
KeyRegistry or the graph merger — not from this reconstruction.

Typical use: migrate a ConsolidationLoop from ``simulate`` mode to ``train``
mode by extracting the embedded knowledge graph from the adapter weights
rather than copying on-disk ``keyed_pairs.json`` sidecars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from paramem.models.loader import switch_adapter
from paramem.training.quadruple_memory import probe_quad

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionResult:
    """Outcome of probing every active key from the adapter weights.

    Attributes:
        graph: Fresh ``nx.MultiDiGraph``.  Each successfully recalled quad
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

    Iterates ``loop.indexed_key_registry.list_active()``, groups keys by
    their ``adapter_id``, calls ``switch_adapter`` once per group, probes
    every key in that group via :func:`~paramem.training.quadruple_memory.probe_quad`,
    and merges the quads into a fresh ``nx.MultiDiGraph``.

    The function is read-only on ``loop.model``: after all probes complete it
    restores the adapter that was active before the first switch.

    Gradient checkpointing is disabled around the probing loop and re-enabled
    in a ``try/finally`` — HF silently disables the KV cache when
    checkpointing is active (CLAUDE.md: applies to ANY ``model.generate()``
    site).

    Args:
        loop: A ``ConsolidationLoop`` instance.  Must have
            ``_indexed_format == "quad"``; QA mode raises
            :exc:`NotImplementedError`.  Reads:

            - ``loop.model`` — PEFT model (adapter switches applied in-place).
            - ``loop.tokenizer`` — tokenizer matching the model.
            - ``loop.indexed_key_registry`` — :class:`~paramem.training.key_registry.KeyRegistry`
              that maps active keys to adapter IDs.
            - ``loop.{episodic,semantic,procedural}_simhash`` — per-adapter
              SimHash registry dicts (may be ``None`` or absent).

        tier: If set (``"episodic"`` | ``"semantic"`` | ``"procedural"``),
            only probe keys whose ``registry.get_adapter_id(key) == tier``.
            ``None`` reconstructs all active keys across all adapters.
        strict: When ``True`` (default), raise :exc:`ReconstructionError` if
            any key failed to recall.  When ``False``, record failures in
            ``ReconstructionResult.failures`` and continue.

    Returns:
        :class:`ReconstructionResult` with:

        - ``graph`` — fresh ``nx.MultiDiGraph``; each successful quad becomes
          an edge ``(subject) -> (object)`` with edge data ``{"key": str,
          "predicate": str}``.
        - ``failures`` — list of failed-probe dicts (empty when all keys
          succeed).

    Raises:
        NotImplementedError: When ``loop._indexed_format != "quad"``.
        ReconstructionError: When ``strict=True`` and at least one key failed.

    Contract:
        Every key in ``registry.list_active()`` (filtered by ``tier``) is
        either represented in ``graph`` as a successful edge or listed in
        ``failures``.  No key is silently dropped.
    """
    _indexed_format = getattr(loop, "_indexed_format", "qa")
    if _indexed_format != "quad":
        raise NotImplementedError(
            f"reconstruct_graph supports indexed_format='quad' only; got {_indexed_format!r}"
        )

    registry = loop.indexed_key_registry
    active_keys: list[str] = registry.list_active()

    # Apply tier filter before building the groups.
    if tier is not None:
        active_keys = [k for k in active_keys if registry.get_adapter_id(k) == tier]

    if not active_keys:
        logger.debug("reconstruct_graph: no active keys to probe (tier=%r)", tier)
        return ReconstructionResult(graph=nx.MultiDiGraph(), failures=[])

    # Group keys by adapter_id so we switch only once per adapter.
    keys_by_adapter: dict[str, list[str]] = {}
    for key in active_keys:
        adapter_id = registry.get_adapter_id(key)
        keys_by_adapter.setdefault(adapter_id, []).append(key)

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

    # Disable gradient checkpointing around all probing — HF silently disables
    # the KV cache when checkpointing is active, which causes silent generation
    # degradation (CLAUDE.md rule applies to ANY model.generate() site).
    model.gradient_checkpointing_disable()
    try:
        for adapter_id, keys in keys_by_adapter.items():
            # Per-adapter SimHash registry: ``{tier}_simhash`` attribute on
            # the loop.  Interim adapter IDs (``episodic_interim_<stamp>``)
            # don't have a dedicated simhash; we pass None (probe_quad
            # defaults confidence to 1.0 when registry is None).
            simhash_registry = getattr(loop, f"{adapter_id}_simhash", None)

            logger.debug(
                "reconstruct_graph: switching to adapter %r, probing %d keys",
                adapter_id,
                len(keys),
            )
            switch_adapter(model, adapter_id)

            for key in keys:
                result = probe_quad(
                    model,
                    tokenizer,
                    key,
                    registry=simhash_registry,
                )

                if "failure_reason" in result:
                    logger.debug(
                        "reconstruct_graph: key %r failed (%s)",
                        key,
                        result["failure_reason"],
                    )
                    failures.append(
                        {
                            "key": key,
                            "adapter_id": adapter_id,
                            "raw_output": result.get("raw_output", ""),
                            "failure_reason": result["failure_reason"],
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
                    # round-trips through ``paramem.server.simulate_store``
                    # (NetworkX reserves the JSON field ``"key"`` for the
                    # multigraph edge identifier; using ``"key"`` here would be
                    # silently clobbered on save→load).
                    from paramem.server.simulate_store import _IK_KEY_ATTR

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
        training_config = getattr(loop, "training_config", None)
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
