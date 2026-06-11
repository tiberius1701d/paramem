"""On-disk persistence for the indexed-key memory layer.

The on-disk medium is a per-tier NetworkX ``MultiDiGraph`` serialised via
``nx.node_link_data`` → JSON → optional age-encrypt → atomic write.  The wire
format is identical to the cumulative knowledge graph written by
:mod:`paramem.graph.merger` so existing encryption infrastructure is reused
without modification.

Public API
----------
- :func:`save_memory_to_disk` — atomic encrypted write.
- :func:`load_memory_from_disk` — decryption-aware read; empty graph on miss.
- :func:`iter_entries` — yield entry dicts for every edge carrying an indexed key.
- :func:`entry_by_key` — look up a single key; ``None`` on miss.
- :func:`keys_for_entity` — edge keys where lower(subject) or lower(object)
  matches the query.
- :func:`keys_for_speaker` — edge keys where ``speaker_id`` matches.
- :func:`build_tier_graph_from_store` — project a :class:`MemoryStore` tier
  to a fresh ``MultiDiGraph`` for persistence.
- :func:`commit_tier_slot` — atomic write of one interim tier slot (registry written last
  as commit signal); mode-switches between adapter-weight venue (train) and graph-JSON venue
  (simulate).

Internal edge attribute naming
-------------------------------
NetworkX's ``node_link_data`` serialisation format uses ``"key"`` as the
reserved field name for the multigraph edge-key integer in the JSON output.
To avoid collision, the indexed-memory key string is stored as the internal
edge-data attribute ``"ik_key"``.  All public API functions
(:func:`iter_entries`, :func:`entry_by_key`, :func:`keys_for_entity`,
:func:`keys_for_speaker`) map ``"ik_key"`` to ``"key"`` in the dict they
return so callers see the canonical entry shape.

The public entry schema is:

    ``key``, ``subject``, ``predicate``, ``object``,
    ``speaker_id``, ``first_seen_cycle``

where ``subject`` and ``object`` are the graph node endpoints and the
remaining four fields come from edge attributes.  ``entry`` is the
shape-agnostic term for "one keyed record" — if the schema grows fields,
the on-disk format accommodates them as additional edge attributes
without rename.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from typing import Literal

    from paramem.training.consolidation import ConsolidationLoop

logger = logging.getLogger(__name__)

# Internal attribute name for the indexed-memory key on graph edges.
# Must not be "key" — that is the NetworkX reserved multigraph edge-key
# field in node_link_data JSON output and would be lost on round-trip.
_IK_KEY_ATTR = "ik_key"

# Internal attribute name for edge provenance (e.g. ``"graph_enrichment"``).
# Must not be "source" — that is the NetworkX reserved edge-source-NODE field
# in node_link_data JSON output; an edge attribute named "source" is silently
# overwritten by the source node's name on save and lost on round-trip (same
# reserved-key collision class as "key" → "ik_key").
_EDGE_SOURCE_ATTR = "edge_source"


def save_memory_to_disk(graph: nx.MultiDiGraph, path: Path, *, encrypted: bool = True) -> None:
    """Atomic encrypted write of *graph* to *path*.

    Serialises via ``nx.node_link_data`` → JSON → bytes, then delegates to
    the infrastructure envelope so the result is age-encrypted when a daily
    identity is loaded, and plaintext otherwise.  The write is atomic:
    ``<path>.tmp`` is written, fsynced, and renamed in a single step so a
    crash leaves no partial file.

    The indexed-memory key is stored as the ``"ik_key"`` edge attribute to
    avoid collision with the NetworkX-reserved ``"key"`` serialisation field.
    :func:`iter_entries` and friends map ``"ik_key"`` back to ``"key"`` in
    the public-facing entry dicts.

    Args:
        graph: The ``MultiDiGraph`` to persist.
        path: Destination path (e.g. ``adapter_dir/episodic/graph.json``).
            Parent directory is created if absent.
        encrypted: When ``True`` (default) routes through
            :func:`paramem.backup.encryption.write_infra_bytes` — age-encrypted
            when a daily identity is loaded, plaintext otherwise.  When
            ``False`` always writes plaintext; reserved for debug-directory
            writers so output is inspectable with ``cat``/``grep``.
    """
    from paramem.backup.encryption import write_infra_bytes, write_plaintext_atomic

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    payload = json.dumps(data, indent=2).encode("utf-8")
    if encrypted:
        write_infra_bytes(path, payload)
    else:
        write_plaintext_atomic(path, payload)
    logger.debug("Memory graph saved to %s (%d edges)", path, graph.number_of_edges())


def load_memory_from_disk(path: Path) -> nx.MultiDiGraph:
    """Decryption-aware load of a memory graph from *path*.

    Returns an empty ``MultiDiGraph()`` when *path* does not exist so callers
    on the boot / inference path never need to guard against a missing file
    explicitly.

    Args:
        path: Source path for the ``graph.json`` file.

    Returns:
        Loaded (or freshly empty) ``nx.MultiDiGraph``.

    Raises:
        json.JSONDecodeError: When the file contains invalid JSON.
        RuntimeError: When the file is age-encrypted but no daily identity is
            loaded.
        OSError: On filesystem errors other than a missing file.
        UnicodeDecodeError: When the file bytes cannot be decoded as UTF-8.
    """
    from paramem.backup.encryption import read_maybe_encrypted

    path = Path(path)
    if not path.exists():
        logger.debug("No memory graph at %s — returning empty MultiDiGraph", path)
        return nx.MultiDiGraph()

    raw = read_maybe_encrypted(path)
    data = json.loads(raw.decode("utf-8"))
    graph = nx.node_link_graph(data, multigraph=True, directed=True)
    logger.debug(
        "Memory graph loaded from %s: %d nodes, %d edges",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def iter_entries(graph: nx.MultiDiGraph) -> Iterator[dict]:
    """Yield entry dicts for every edge in *graph* that carries an indexed-memory key.

    Edges without the ``"ik_key"`` internal attribute (e.g. cumulative-graph
    edges that pre-date key assignment) are silently skipped so this function
    is safe to call on mixed-content graphs.

    Each yielded dict has exactly the canonical entry fields:

        ``key``, ``subject``, ``predicate``, ``object``,
        ``speaker_id``, ``first_seen_cycle``

    where ``subject`` and ``object`` are taken from the graph topology (the
    source and target node names of the edge) and ``key`` is mapped from the
    internal ``"ik_key"`` edge attribute.

    Note: the ``keys=True`` positional argument in ``graph.edges(keys=True,
    data=True)`` refers to NetworkX's internal multigraph edge-key integer,
    **not** our indexed-memory key attribute.

    Args:
        graph: Source ``MultiDiGraph``.

    Yields:
        Entry dicts with the canonical schema fields.
    """
    for subject, object_, _nx_edge_key, data in graph.edges(keys=True, data=True):
        if _IK_KEY_ATTR not in data:
            continue
        yield {
            "key": data[_IK_KEY_ATTR],
            "subject": subject,
            "predicate": data.get("predicate", ""),
            "object": object_,
            "speaker_id": data.get("speaker_id", ""),
            "first_seen_cycle": data.get("first_seen_cycle", 0),
        }


def entry_by_key(graph: nx.MultiDiGraph, key: str) -> dict | None:
    """Return the entry dict for *key*, or ``None`` when the key is absent.

    Linear scan over all edges (graphs are per-tier, expected to contain at
    most a few hundred edges).  Returns the first matching edge without
    examining further edges, so the function is ``O(n)`` worst-case.

    Args:
        graph: Source ``MultiDiGraph``.
        key: The indexed-memory key to look up (e.g. ``"graph1"``).

    Returns:
        Entry dict with canonical schema fields on hit; ``None`` on miss.
    """
    for entry in iter_entries(graph):
        if entry["key"] == key:
            return entry
    return None


def keys_for_entity(graph: nx.MultiDiGraph, entity_lower: str) -> set[str]:
    """Return every indexed-memory key whose subject or object matches *entity_lower*.

    Case-insensitive: ``entity_lower`` must already be lowercased by the
    caller; the function lowercases the subject/object read from each edge.

    Args:
        graph: Source ``MultiDiGraph``.
        entity_lower: Lowercase entity name to match (e.g. ``"alice"``).

    Returns:
        Set of key strings where the edge subject or object matches; may be
        empty.
    """
    matched: set[str] = set()
    for entry in iter_entries(graph):
        if entry["subject"].lower() == entity_lower or entry["object"].lower() == entity_lower:
            matched.add(entry["key"])
    return matched


def keys_for_speaker(graph: nx.MultiDiGraph, speaker_id: str) -> set[str]:
    """Return every indexed-memory key whose ``speaker_id`` matches *speaker_id*.

    Exact string match (not case-insensitive) to preserve speaker ID
    semantics — speaker IDs are controlled identifiers, not free-form names.

    Args:
        graph: Source ``MultiDiGraph``.
        speaker_id: Speaker ID to match (e.g. ``"Speaker0"``).

    Returns:
        Set of key strings where the edge ``speaker_id`` matches; may be
        empty.
    """
    matched: set[str] = set()
    for entry in iter_entries(graph):
        if entry["speaker_id"] == speaker_id:
            matched.add(entry["key"])
    return matched


def _add_keyed_edge(
    graph: nx.MultiDiGraph,
    subject: str,
    object_: str,
    *,
    indexed_key: str,
    predicate: str,
    speaker_id: str,
    first_seen_cycle: int,
) -> None:
    """Add an edge to *graph* with the indexed-memory key in the ``"ik_key"`` attribute.

    The indexed-memory key is stored under the ``"ik_key"`` attribute (not
    ``"key"``) because NetworkX's ``node_link_data`` serialisation uses
    ``"key"`` as its own reserved field for the multigraph edge-key integer.
    Storing under ``"key"`` would cause the value to be lost on round-trip.

    :func:`iter_entries` and friends map ``"ik_key"`` back to ``"key"`` so
    callers see the canonical entry shape.

    Args:
        graph: Target ``MultiDiGraph``.
        subject: Source node (subject entity).
        object_: Target node (object entity).
        indexed_key: The indexed-memory key string (e.g. ``"graph1"``).
        predicate: Relation predicate.
        speaker_id: Speaker scope.
        first_seen_cycle: Cycle count at first insertion.
    """
    graph.add_edge(
        subject,
        object_,
        **{
            _IK_KEY_ATTR: indexed_key,
            "predicate": predicate,
            "speaker_id": speaker_id,
            "first_seen_cycle": first_seen_cycle,
        },
    )


def build_tier_graph_from_store(store, tier: str) -> nx.MultiDiGraph:
    """Project the *tier* slice of a :class:`MemoryStore` into a fresh ``MultiDiGraph``.

    Iterates every key in ``store.simhashes_in_tier(tier)`` — keeping the
    simhash dict as the enumeration spine so replay-disabled stores work
    correctly — but **skips stale keys** so the projected graph is active-only.
    Stale simhash entries are retained on disk (step-6 wholesale persist in
    ``commit_tier_slot`` is unchanged); they are excluded only from the graph
    projection so superseded facts are never re-materialized into ``graph.json``.

    Reads the matching entry from the store to add an edge
    ``(subject → object)`` with edge-data
    ``{ik_key, predicate, speaker_id, first_seen_cycle}`` (the indexed-memory
    key is stored as ``"ik_key"`` to avoid the NetworkX ``"key"`` collision;
    :func:`iter_entries` maps it back to ``"key"`` for callers).

    The caller is responsible for persisting the returned graph with
    :func:`save_memory_to_disk`.

    Args:
        store: A :class:`paramem.memory.store.MemoryStore`
            (or any object exposing ``simhashes_in_tier(tier) -> dict[str,int]``,
            ``get(key) -> dict | None``, and ``is_stale(key) -> bool``).
        tier: One of ``"episodic"``, ``"semantic"``, or ``"procedural"``.

    Returns:
        A new ``nx.MultiDiGraph`` with one edge per *active* key in the tier's
        SimHash registry (stale keys excluded).

    Raises:
        KeyError: When an active key present in ``simhashes_in_tier`` is absent
            from the store's entry cache.  Stale keys never raise (they are
            skipped before the entry lookup).  A simhash/entry divergence on an
            active key is a data-integrity bug to surface, not paper over.
    """
    simhash_registry: dict[str, int] = store.simhashes_in_tier(tier)
    graph = nx.MultiDiGraph()
    for indexed_key in simhash_registry:
        # Skip stale keys — their simhash is retained on disk for the
        # stale-echo seam, but they must not be projected into graph.json
        # (superseded facts must not re-materialize).
        if store.is_stale(indexed_key):
            continue
        entry = store.get(indexed_key)
        if entry is None:
            raise KeyError(indexed_key)
        _add_keyed_edge(
            graph,
            entry["subject"],
            entry["object"],
            indexed_key=indexed_key,
            predicate=entry.get("predicate", ""),
            speaker_id=entry.get("speaker_id", ""),
            first_seen_cycle=entry.get("first_seen_cycle", 0),
        )
    return graph


# ---------------------------------------------------------------------------
# Registry I/O and lifecycle helpers
# (relocated from paramem.training.indexed_memory on 2026-05-20)
# ---------------------------------------------------------------------------


def save_registry(registry: dict, path) -> None:
    """Save registry to a JSON file — envelope-encrypted when a master key is
    set, plaintext otherwise.  Handles both simple and enriched formats.

    Args:
        registry: Dict mapping key → fingerprint (simple) or key → metadata
            dict (enriched).
        path: Destination path (``str`` or :class:`pathlib.Path`).
    """
    from pathlib import Path as _Path

    from paramem.backup.encryption import write_infra_bytes

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry, indent=2).encode("utf-8")
    write_infra_bytes(path, payload)


def load_registry(path) -> dict:
    """Load registry from a JSON file — transparently decrypts age-wrapped
    content when the daily identity is loaded.  Handles both simple and
    enriched formats.

    Args:
        path: Source path (``str`` or :class:`pathlib.Path`).

    Returns:
        Registry dict mapping key → fingerprint or key → metadata dict.
    """
    from pathlib import Path as _Path

    from paramem.backup.encryption import read_maybe_encrypted

    return json.loads(read_maybe_encrypted(_Path(path)).decode("utf-8"))


def get_active_keys(registry: dict) -> list[str]:
    """Return only active (non-stale) keys from an enriched registry.

    Also handles the simple format (key → int) for backward compatibility.

    Args:
        registry: Either a simple ``{key: int}`` registry or an enriched
            ``{key: {"status": str, ...}}`` registry.

    Returns:
        Sorted list of active key strings.
    """
    active = []
    for key, entry in registry.items():
        if isinstance(entry, int):
            active.append(key)
        elif isinstance(entry, dict) and entry.get("status", "active") == "active":
            active.append(key)
    return sorted(active)


def commit_tier_slot(
    *,
    loop: "ConsolidationLoop",
    tier: str,
    adapter_name: str,
    stamp: str,
    mode: "Literal['simulate', 'train']",
    all_keyed: "list[dict]",
    output_dir: Path,
) -> None:
    """Atomic write of one interim-tier slot, venue-switching on mode.

    Registry is written last as the commit signal: its presence on disk means
    all preceding files (weights, simhash) are complete.

    Executes the full atomic commit sequence for a single tier produced by
    :meth:`paramem.training.consolidation.ConsolidationLoop.run_consolidation_cycle`:

    1. ``loop.store.registry(tier).save_bytes()`` — serialise the tier registry
       to canonical UTF-8 JSON bytes (no disk I/O).
    2. ``hashlib.sha256(payload)`` — hash the bytes so the manifest can stamp
       them before the registry is written to disk (pre-stamp invariant:
       manifest records the hash before the registry file exists).
    3. Build manifest (train mode only) — calls
       :func:`paramem.adapters.manifest.build_manifest_for` with the
       ``registry_sha256_override`` and ``window_stamp=stamp``.  Simulate mode
       has no live PEFT model to hash; the manifest step is skipped.
    4. Determine slot root via
       :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`.
       Same root for both modes.
    5. Write venue payload:

       - **Train mode**: :func:`paramem.models.loader.atomic_save_adapter`
         writes the PEFT adapter weights into a timestamped slot dir under the
         root.  The manifest (from step 3) is embedded as ``meta.json``.
       - **Simulate mode**: builds a ``MultiDiGraph`` from *all_keyed* and
         writes it as ``<slot_root>/graph.json`` via :func:`save_memory_to_disk`
         (encrypted/plaintext depending on daily-key state).  No PEFT weights
         are written.

    6. Write simhash registry to ``<slot_root>/simhash_registry.json`` (both
       modes) via :func:`paramem.memory.persistence.save_registry`.
    7. Flush the exact registry bytes from step 1 to
       ``<slot_root>/indexed_key_registry.json`` as the commit signal (both
       modes) via :meth:`paramem.training.key_registry.KeyRegistry.save_from_bytes`.
       This is the last write — its presence on disk signals that all preceding
       files are complete.

    Crash semantics: a kill after step 5 but before step 7 leaves the slot
    present without the registry file.  The boot validator in
    :func:`paramem.server.app._mount_adapters_from_slots` (via
    :func:`paramem.server.app.find_live_slot` / ``iter_interim_dirs``) skips
    any interim dir whose ``indexed_key_registry.json`` is absent and whose
    slot contains neither ``adapter_model.safetensors`` (train) nor
    ``graph.json`` (simulate).  A partial slot — payload written but registry
    absent — is silently ignored on restart; the slot becomes eligible for
    wipe by the boot validator's torn-write cleanup pass.

    Args:
        loop: The live :class:`paramem.training.consolidation.ConsolidationLoop`
            whose ``model``, ``tokenizer``, ``store``, and ``fingerprint_cache``
            are used for the write.
        tier: Tier name for registry lookup (e.g. ``"episodic"`` or
            ``"procedural"``).  Must be a tier registered in ``loop.store``.
        adapter_name: PEFT adapter name (e.g. ``"episodic_interim_YYYYMMDDTHHMM"``
            or ``"procedural"``).  Determines the on-disk slot root via
            :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`.
        stamp: Sub-interval stamp (``"YYYYMMDDTHHMM"``) used as the
            ``window_stamp`` in the manifest and forwarded to
            :func:`build_manifest_for`.
        mode: ``"train"`` writes adapter weights; ``"simulate"`` writes
            ``graph.json`` sidecar instead.
        all_keyed: The full list of keyed-pair dicts trained into (or
            simulated for) this tier slot.  Used by the simulate path to build
            the ``MultiDiGraph`` from the ground-truth list rather than by
            scanning the full store (avoids including keys from other tiers).
        output_dir: Adapter store root (``loop.output_dir``).  Slot root is
            derived from this via ``adapter_slot_root_for_name``.

    Returns:
        ``None``.  Raises on I/O failure.

    Raises:
        RuntimeError: When ``loop.store.replay_enabled`` is ``False`` (no
            registry to commit).
        KeyError: When ``tier`` has no registry in ``loop.store``.
    """
    import hashlib as _hashlib

    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    if not loop.store.replay_enabled:
        raise RuntimeError("commit_tier_slot: replay is disabled — no registry to commit")

    # Store keys are written under the adapter NAME (per
    # MemoryStore.put(tier=adapter_name, ...) in run_consolidation_cycle), so
    # registry / simhash / graph projections must read under the same key.
    # `tier` is retained for log messages and the on-disk slot hierarchy via
    # adapter_slot_root_for_name (which dispatches by adapter_name internally).
    tier_reg = loop.store.registry(adapter_name)
    if tier_reg is None:
        raise KeyError(f"commit_tier_slot: no registry for adapter {adapter_name!r}")

    # --- Step 1+2: Serialize and hash (no disk I/O) ---
    payload = tier_reg.save_bytes()
    registry_sha256 = _hashlib.sha256(payload).hexdigest()

    slot_root = adapter_slot_root_for_name(output_dir, adapter_name)
    slot_root.mkdir(parents=True, exist_ok=True)

    # Track whether the registry flush (step 7 — the commit signal) has
    # completed.  Used in the finally block to remove the orphan slot dir on
    # any exception that fires before the flush.  After the flush the slot is
    # "live"; exceptions thereafter leave it intact, which is the documented
    # crash-safe outcome.
    _registry_flushed = False
    try:
        # --- Step 5: Write venue payload ---
        if mode == "train":
            # Build manifest stamped with the pre-hashed registry sha256.
            # A manifest failure is a load-bearing bug — the slot becomes
            # unmountable on boot because find_live_slot cannot match the
            # registry hash.  Let the exception propagate so the caller's error
            # path is triggered and the session is retried rather than silently
            # written with a missing manifest.
            from paramem.adapters.manifest import build_manifest_for as _build_manifest_for

            fingerprint_cache = getattr(loop, "fingerprint_cache", None)
            manifest = _build_manifest_for(
                loop.model,
                loop.tokenizer,
                adapter_name,
                registry_path=None,
                key_count=len(loop.store.all_active_keys()),
                base_model_hash_cache=fingerprint_cache,
                registry_sha256_override=registry_sha256,
                window_stamp=stamp,
                adapter_root=output_dir,
            )

            # Use save_adapter (thin forwarder to atomic_save_adapter) so tests can
            # patch paramem.models.loader.save_adapter without reaching atomic internals.
            from paramem.models.loader import save_adapter as _save_adapter

            _save_adapter(loop.model, slot_root, adapter_name, manifest=manifest)
            logger.debug(
                "commit_tier_slot: adapter weights saved for %s (tier=%s)",
                adapter_name,
                tier,
            )
        else:
            # Simulate mode: build graph from all_keyed and write graph.json.
            # When all_keyed is empty but the store has entries for the tier
            # (e.g. procedural, whose mutations are applied directly in
            # _prepare_procedural_keys_for_tier rather than via the caller-
            # supplied all_keyed list), re-project from the canonical store so
            # we do not overwrite prior content with an empty graph.
            if all_keyed:
                graph = nx.MultiDiGraph()
                for kp in all_keyed:
                    _add_keyed_edge(
                        graph,
                        kp["subject"],
                        kp["object"],
                        indexed_key=kp["key"],
                        predicate=kp.get("predicate", ""),
                        speaker_id=kp.get("speaker_id", ""),
                        first_seen_cycle=kp.get("first_seen_cycle", 0),
                    )
            else:
                # Caller passed [] — re-project from the canonical store to
                # capture keys written directly by _prepare_procedural_keys_for_tier
                # (which writes to loop.store in simulate mode instead of
                # deferring to the caller-supplied list as the episodic path does).
                graph = build_tier_graph_from_store(loop.store, adapter_name)
            graph_path = slot_root / "graph.json"
            save_memory_to_disk(graph, graph_path)
            logger.debug(
                "commit_tier_slot: graph.json written for %s (tier=%s, %d edges)",
                adapter_name,
                tier,
                graph.number_of_edges(),
            )

        # --- Step 6: SimHash registry (both modes) ---
        save_registry(
            loop.store.simhashes_in_tier(adapter_name),
            slot_root / "simhash_registry.json",
        )

        # --- Step 7: Registry flush — commit signal (both modes, LAST write) ---
        # After this returns the slot is live on disk.  Any exception before
        # this point is caught by the finally block below which removes the
        # orphan slot dir.
        tier_reg.save_from_bytes(
            payload,
            slot_root / "indexed_key_registry.json",
            consolidating=True,
        )
        _registry_flushed = True
    finally:
        if not _registry_flushed:
            # The commit signal was never written.  Remove the orphan slot dir
            # so the boot validator does not encounter partial content.
            # Best-effort: cleanup errors are swallowed (ignore_errors=True) so
            # the original exception propagates unmodified to the caller.
            import shutil as _shutil

            _shutil.rmtree(slot_root, ignore_errors=True)
    logger.info(
        "commit_tier_slot: committed %s (tier=%s, mode=%s, keys=%d)",
        adapter_name,
        tier,
        mode,
        len(tier_reg),
    )
