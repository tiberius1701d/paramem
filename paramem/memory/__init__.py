"""Indexed-key memory layer: entries, store, source, persistence, probe, interim adapters.

Submodules are imported here at package scope.  Callers that import from
the package root (``from paramem.memory import MemoryStore``) are supported,
but existing callers using ``from paramem.memory.<submodule> import X`` do
not need to change.

Cycle note: submodules cross-reference each other by full module path
(``from paramem.memory.entry import ...``), which does NOT re-trigger this
``__init__``.  Do NOT add ``from paramem.memory import X`` inside any
submodule — that would create an import-time cycle through this file.
"""

from paramem.memory.entry import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    RECALL_TEMPLATE,
    SIMHASH_BITS,
    assign_keys,
    build_registry,
    compute_simhash,
    entry_fact_text,
    format_entry_training,
    get_active_keys,
    get_simhash,
    load_registry,
    parse_recalled_entry,
    save_registry,
    simhash_confidence,
    verify_confidence,
)
from paramem.memory.interim_adapter import (
    INTERIM_NAME_PREFIX,
    adapter_slot_root_for_name,
    create_interim_adapter,
    current_full_consolidation_stamp,
    current_interim_stamp,
    iter_interim_dirs,
    unload_interim_adapters,
)
from paramem.memory.persistence import (
    commit_tier_slot,
    entry_by_key,
    iter_entries,
    keys_for_entity,
    keys_for_speaker,
    load_memory_from_disk,
    save_memory_to_disk,
)
from paramem.memory.probe import probe_keys_grouped_by_adapter
from paramem.memory.source import DiskMemorySource, MemorySource, WeightMemorySource
from paramem.memory.store import MemoryStore

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "INTERIM_NAME_PREFIX",
    "RECALL_TEMPLATE",
    "SIMHASH_BITS",
    "DiskMemorySource",
    "MemorySource",
    "MemoryStore",
    "WeightMemorySource",
    "adapter_slot_root_for_name",
    "assign_keys",
    "build_registry",
    "commit_tier_slot",
    "compute_simhash",
    "create_interim_adapter",
    "current_full_consolidation_stamp",
    "current_interim_stamp",
    "entry_by_key",
    "entry_fact_text",
    "format_entry_training",
    "get_active_keys",
    "get_simhash",
    "iter_entries",
    "iter_interim_dirs",
    "keys_for_entity",
    "keys_for_speaker",
    "load_memory_from_disk",
    "load_registry",
    "parse_recalled_entry",
    "probe_keys_grouped_by_adapter",
    "save_memory_to_disk",
    "save_registry",
    "simhash_confidence",
    "unload_interim_adapters",
    "verify_confidence",
]
