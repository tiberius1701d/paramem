"""Source-of-truth abstraction for the indexed-key memory layer.

A :class:`MemorySource` retrieves indexed-key entries from the authoritative
medium for the current consolidation mode.  Two implementations exist:

* :class:`WeightMemorySource` — train mode.  Probes adapter weights via the
  recall template and reconstructs the entry from generated output.
* :class:`DiskMemorySource` — simulate mode.  Reads encrypted per-tier
  ``graph.json`` files and decodes the entry directly.

Both implementations return the same canonical entry shape so callers
(boot hydration, on-miss inference probe, fold hydration, active-store
migration) are mode-agnostic.  :func:`build_memory_source` is the ONE
construction site — callers name the mode, never the class.

The source is **not** the cache — :class:`paramem.memory.store.MemoryStore`
is.  A source is invoked at boot to populate the cache and on cache miss to
materialise individual entries.  The cache and the source together form the
inference read path; either alone is incomplete (boot-cached but stale on a
cycle, or sourced fresh but expensive per query).

Naming
------
``entry`` is the shape-agnostic term for "one keyed record" and is
content-only: ``{key, subject, predicate, object}`` plus a source's own
derived fields (``fact_text``, ``raw_output``, and — for
:class:`WeightMemorySource` only — a real SimHash-verified ``confidence``).
No source emits ``speaker_id`` or a fabricated confidence.  Speaker
attribution lives exclusively in
:attr:`~paramem.memory.store.MemoryStore._bookkeeping`, written by
consolidation at fold time (never derived from a probe result), and is read
back by :func:`~paramem.memory.persistence.build_tier_graph_from_store` when
re-persisting a tier graph.  The store-boundary SimHash gate in
:meth:`~paramem.memory.store.MemoryStore.probe` is the sole confidence
authority for both cache hits and source-served results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable

from paramem.memory.entry import DEFAULT_CONFIDENCE_THRESHOLD


@runtime_checkable
class MemorySource(Protocol):
    """Read-side contract for the indexed-key source of truth.

    Implementations resolve a batch of (adapter, key) pairs into a flat
    ``{key → entry-or-None}`` mapping.  Adapter ordering in
    *keys_by_adapter* is preserved through the result so callers can rely
    on the router's preferred probe order (procedural → episodic →
    semantic → newest-interim) reaching the model in that order.

    Each hit carries the canonical entry fields documented in
    :func:`paramem.memory.probe.probe_keys_grouped_by_adapter`.
    Misses (unknown key, decoding failure, missing adapter) map to ``None``.
    """

    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, dict | None]:  # pragma: no cover — Protocol
        ...


class WeightMemorySource:
    """Train-mode source.  Materialises entries by probing adapter weights.

    Wraps :func:`probe_keys_grouped_by_adapter`.  The wrapped function does
    one ``switch_adapter`` per group and ``batch_size`` keys per
    ``model.generate`` call, so the per-call cost scales linearly with the
    total key count divided by ``batch_size``.

    The model, tokenizer, and per-adapter format mapping are captured at
    construction so callers don't thread them through every call.  When the
    set of mounted adapters or their formats changes (e.g. after a
    consolidation cycle finalize) the lifespan rebuilds the source.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        registry: dict[str, int] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_new_tokens: int = 200,
        batch_size: int,
    ) -> None:
        """Initialise the weight-based memory source.

        Args:
            model: PeftModel already loaded in memory.
            tokenizer: Tokenizer matching the model.
            registry: Optional SimHash registry for confidence verification.
            confidence_threshold: Minimum confidence to accept a recalled entry.
            max_new_tokens: Maximum tokens to generate per probe.
            batch_size: Number of keys per ``model.generate`` call.  MUST be
                supplied from ``config.consolidation.recall_probe_batch_size``
                — no default is provided so callers cannot silently fall back
                to single-key generation.
        """
        # BASE-MODEL HOLDER (WeightMemorySource): every construction goes
        # through build_memory_source, and every caller of that keeps the result
        # as a frame-local it drops before returning —
        # _release_base_model_in_process cannot reach a caller-frame local.
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, dict | None]:
        """Probe adapter weights for the given keys.

        Args:
            keys_by_adapter: Ordered mapping of adapter name → list of keys.
            should_abort: Optional callable forwarded to
                :func:`~paramem.memory.probe.probe_keys_grouped_by_adapter`.
                When it returns ``True`` before an adapter group starts, the
                probe exits early with partial results — yielding the GPU to
                a waiting ``/chat`` request.

        Returns:
            Flat ``{key → result | None}`` mapping.  May be partial when
            ``should_abort`` fires.
        """
        # Lazy import so test monkeypatches against
        # ``paramem.memory.probe.probe_keys_grouped_by_adapter``
        # take effect without re-binding through this module.
        from paramem.memory.probe import probe_keys_grouped_by_adapter

        return probe_keys_grouped_by_adapter(
            self.model,
            self.tokenizer,
            keys_by_adapter,
            max_new_tokens=self.max_new_tokens,
            registry=self.registry,
            confidence_threshold=self.confidence_threshold,
            batch_size=self.batch_size,
            should_abort=should_abort,
        )


class DiskMemorySource:
    """Simulate-mode source.  Materialises entries by reading per-tier graph.json.

    No model interaction, no GPU, no switch_adapter — pure disk read +
    JSON decode.  Per-call cost scales with the per-tier graph size (a
    few hundred edges typically).

    Path resolution uses :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`
    so both main tiers (``"episodic"``, ``"semantic"``, ``"procedural"`` — flat
    under ``<store_dir>/<tier>/``) and interim adapters
    (``"episodic_interim_<stamp>"`` — nested under
    ``<store_dir>/episodic/interim_<stamp>/``) resolve correctly.

    *store_dir* is the adapter root (``config.adapter_dir``) — the same
    directory that ``commit_tier_slot`` writes graph.json into.
    """

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)

    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
        should_abort: Callable[[], bool] | None = None,  # noqa: ARG002 — CPU path, no abort needed
    ) -> dict[str, dict | None]:
        """Read entries from per-tier graph.json files on disk.

        Args:
            keys_by_adapter: Ordered mapping of adapter name → list of keys.
            should_abort: Accepted for interface parity with
                :class:`WeightMemorySource`; ignored here because the disk
                path is CPU-bound and fast (no GPU contention).

        Returns:
            Flat ``{key → result | None}`` mapping.
        """
        import json

        from paramem.memory.entry import entry_fact_text
        from paramem.memory.interim_adapter import adapter_slot_root_for_name
        from paramem.memory.persistence import (
            entry_by_key,
            load_memory_from_disk,
        )

        results: dict[str, dict | None] = {}
        for adapter_name, keys in keys_by_adapter.items():
            if not keys:
                continue
            graph_path = adapter_slot_root_for_name(self.store_dir, adapter_name) / "graph.json"
            graph = load_memory_from_disk(graph_path)
            for key in keys:
                entry = entry_by_key(graph, key)
                if entry is None:
                    results[key] = None
                    continue
                results[key] = {
                    "key": key,
                    "subject": entry.get("subject", ""),
                    "predicate": entry.get("predicate", ""),
                    "object": entry.get("object", ""),
                    "fact_text": entry_fact_text(entry),
                    "raw_output": json.dumps(
                        {
                            "key": key,
                            "subject": entry.get("subject", ""),
                            "predicate": entry.get("predicate", ""),
                            "object": entry.get("object", ""),
                        }
                    ),
                }
        return results


def build_memory_source(
    *,
    mode: "Literal['train', 'simulate']",
    adapter_dir: "Path | str",
    batch_size: int,
    model=None,
    tokenizer=None,
    cached_registry: bool = False,
) -> "MemorySource | None":
    """Construct the :class:`MemorySource` for *mode* — the ONE construction site.

    Every path that needs a source goes through here: boot / post-fold store
    hydration (``app._build_store_contents``), the per-query on-miss probe
    (``inference._probe_and_reason``), and the per-fold store hydration
    (``ConsolidationLoop._hydrate_store_for_fold``).  The mode → class mapping
    exists exactly once, which is why this is the only function in
    ``paramem/memory/`` on the mode-fork allowlist.

    **BASE-MODEL HOLDER** — a returned :class:`WeightMemorySource` captures
    *model*.  The caller owns the lifetime: keep it as a frame-local and drop it
    before returning, never on ``self`` (see the invariant header on
    ``app._release_base_model_in_process``).

    Args:
        mode: Consolidation persistence mode.  ``"simulate"`` → graph.json on
            disk; ``"train"`` → adapter weights.  Production sources:
            ``config.consolidation.mode`` (server sites) and
            ``ConsolidationLoop._venue_from_scope(scope)`` (fold site).
        adapter_dir: Adapter root.  ``config.adapter_dir`` on the server sites,
            ``ConsolidationLoop.output_dir`` in the fold — the same directory
            the per-tier ``graph.json`` and ``indexed_key_registry.json`` files
            are written into.
        batch_size: Keys per ``model.generate`` call for the weight probe.
            Production source: ``config.consolidation.recall_probe_batch_size``
            (server sites) / ``TrainingConfig.recall_probe_batch_size`` (fold),
            which ``ServerConfig`` derives from the same field.  Required even
            in simulate mode so the signature does not fork.
        model: Loaded ``PeftModel``.  Train mode only; ``None`` means no local
            model (cloud-only boot, or a failed load).
        tokenizer: Tokenizer matching *model*.  Train mode only.
        cached_registry: Forwarded to
            :meth:`~paramem.memory.store.MemoryStore.read_simhash_registry_from_disk`
            as its ``cached`` keyword (train mode only; no effect in simulate
            mode, which never reads the simhash registry).  Default ``False``
            re-reads every tier registry from disk on every call — the
            correct choice for hydration callers, which run before
            :meth:`~paramem.server.router.QueryRouter.reload` and must see
            disk truth.  Only the per-turn inference probe
            (``inference._probe_and_reason``) opts in.

    Returns:
        A :class:`DiskMemorySource` in simulate mode; a
        :class:`WeightMemorySource` in train mode; ``None`` in train mode when
        no model is loaded — the caller then has no source of truth and must
        decide whether to skip or degrade.
    """
    if mode == "simulate":
        return DiskMemorySource(adapter_dir)
    if model is None:
        return None

    # SimHash registry is DERIVED from adapter_dir, never passed in: it gates
    # recalled entries before they enter the cache (the store-boundary gate in
    # MemoryStore.probe remains the hermetic authority).
    from paramem.memory.store import MemoryStore

    return WeightMemorySource(
        model,
        tokenizer,
        registry=MemoryStore.read_simhash_registry_from_disk(adapter_dir, cached=cached_registry),
        batch_size=batch_size,
    )
