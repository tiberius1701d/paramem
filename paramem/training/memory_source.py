"""Source-of-truth abstraction for the indexed-key memory layer.

A :class:`MemorySource` retrieves indexed-key entries from the authoritative
medium for the current consolidation mode.  Two implementations exist:

* :class:`WeightMemorySource` — train mode.  Probes adapter weights via the
  recall template and reconstructs the entry from generated output.
* :class:`DiskMemorySource` — simulate mode.  Reads encrypted per-tier
  ``graph.json`` files and decodes the entry directly.

Both implementations return the same canonical entry shape so callers
(boot hydration, on-miss inference probe, in-training verification,
active-store migration) are mode-agnostic.

The source is **not** the cache — :class:`paramem.training.memory_store.MemoryStore`
is.  A source is invoked at boot to populate the cache and on cache miss to
materialise individual entries.  The cache and the source together form the
inference read path; either alone is incomplete (boot-cached but stale on a
cycle, or sourced fresh but expensive per query).

Naming
------
``entry`` is the shape-agnostic term for "one keyed record".  Today the
schema is ``{subject, predicate, object, speaker_id}`` with the canonical
fields ``{key, subject, predicate, object, speaker_id, first_seen_cycle,
confidence, fact_text, raw_output}``.  If the schema evolves the source's
contract still holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from paramem.training.indexed_memory import DEFAULT_CONFIDENCE_THRESHOLD


@runtime_checkable
class MemorySource(Protocol):
    """Read-side contract for the indexed-key source of truth.

    Implementations resolve a batch of (adapter, key) pairs into a flat
    ``{key → entry-or-None}`` mapping.  Adapter ordering in
    *keys_by_adapter* is preserved through the result so callers can rely
    on the router's preferred probe order (procedural → episodic →
    semantic → newest-interim) reaching the model in that order.

    Each hit carries the canonical entry fields documented in
    :func:`paramem.training.indexed_memory.probe_keys_grouped_by_adapter`.
    Misses (unknown key, decoding failure, missing adapter) map to ``None``.
    """

    def probe(
        self, keys_by_adapter: dict[str, list[str]]
    ) -> dict[str, dict | None]:  # pragma: no cover — Protocol
        ...


class WeightMemorySource:
    """Train-mode source.  Materialises entries by probing adapter weights.

    Wraps :func:`probe_keys_grouped_by_adapter`.  The wrapped function does
    one ``switch_adapter`` per group and one ``model.generate`` per key,
    so the per-call cost scales linearly with the total key count.

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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.max_new_tokens = max_new_tokens

    def probe(self, keys_by_adapter: dict[str, list[str]]) -> dict[str, dict | None]:
        # Lazy import so test monkeypatches against
        # ``paramem.training.indexed_memory.probe_keys_grouped_by_adapter``
        # take effect without re-binding through this module.
        from paramem.training.indexed_memory import probe_keys_grouped_by_adapter

        return probe_keys_grouped_by_adapter(
            self.model,
            self.tokenizer,
            keys_by_adapter,
            max_new_tokens=self.max_new_tokens,
            registry=self.registry,
            confidence_threshold=self.confidence_threshold,
        )


class DiskMemorySource:
    """Simulate-mode source.  Materialises entries by reading per-tier graph.json.

    No model interaction, no GPU, no switch_adapter — pure disk read +
    JSON decode.  Per-call cost scales with the per-tier graph size (a
    few hundred edges typically).

    The per-tier directory layout is ``<store_dir>/<adapter_name>/graph.json``
    where *adapter_name* is the tier name verbatim (``"episodic"``,
    ``"semantic"``, ``"procedural"``, or ``"episodic_interim_<stamp>"``).
    """

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)

    def probe(self, keys_by_adapter: dict[str, list[str]]) -> dict[str, dict | None]:
        import json

        from paramem.training.entry_memory import entry_fact_text
        from paramem.training.memory_persistence import (
            entry_by_key,
            load_memory_from_disk,
        )

        results: dict[str, dict | None] = {}
        for adapter_name, keys in keys_by_adapter.items():
            if not keys:
                continue
            graph_path = self.store_dir / adapter_name / "graph.json"
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
                    "speaker_id": entry.get("speaker_id", ""),
                    "first_seen_cycle": entry.get("first_seen_cycle", 0),
                    "confidence": 1.0,
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
