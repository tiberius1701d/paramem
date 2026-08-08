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
- :func:`build_tier_graph_from_store` — project a :class:`MemoryStore` tier
  to a fresh ``MultiDiGraph`` for persistence.
- :func:`erase_keys_from_graph_file` — surgical read-modify-write removal of
  the edges for a set of keys from an on-disk tier graph (used by
  ``POST /speaker/forget`` to retire the underlying fact content, not just
  the registry pointer).
- :func:`erase_keys_and_restamp_manifest` — hard-erases a key set from every
  affected tier's registry, graph, and (for a surviving tier) weight-slot
  manifest in one atomic-ordered sequence; shared by every out-of-fold
  registry-mutation caller (``POST /speaker/forget`` today).
- :func:`commit_tier_slot` — atomic write of one interim tier slot (registry written last
  as commit signal); mode-switches between adapter-weight venue (train) and graph-JSON venue
  (simulate).
- :func:`reap_tier_artifacts` — remove one tier's on-disk artifacts, shape derived
  from the tier root itself (interim slot vs. main tier). Rename-condemns
  each removed root into ``.pending-delete/`` before deleting it there, so a
  crash mid-delete leaves the corpse out of the live namespace instead of
  half-deleted in place.
- :func:`resume_pending_reaps` — boot-time sweep that finishes any deletion
  :func:`reap_tier_artifacts` left stranded under ``.pending-delete/``.

Internal edge attribute naming
-------------------------------
NetworkX's ``node_link_data`` serialisation format uses ``"key"`` as the
reserved field name for the multigraph edge-key integer in the JSON output.
To avoid collision, the indexed-memory key string is stored as the internal
edge-data attribute ``"ik_key"``.  All public API functions
(:func:`iter_entries`, :func:`entry_by_key`) map ``"ik_key"`` to ``"key"``
in the dict they return so callers see the canonical entry shape.

The public entry schema is:

    ``key``, ``subject``, ``predicate``, ``object``, ``speaker_id``

where ``subject`` and ``object`` are the graph node endpoints and the
remaining fields come from edge attributes.  ``entry`` is the
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
    from collections.abc import Callable
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

# Tombstone directory :func:`reap_tier_artifacts` rename-condemns into before
# deleting, and :func:`resume_pending_reaps` sweeps at boot. Dot-prefixed so
# it is invisible to every tier/interim-dir enumerator (``interim_*`` globs,
# ``find_live_slot``'s dot-entry skip) without those callers needing to know
# it exists. Lives at the adapter-dir root, one dir shared by both the
# interim-slot and main-tier reap branches.
_PENDING_DELETE_DIR_NAME = ".pending-delete"


def save_memory_to_disk(graph: nx.MultiDiGraph, path: Path) -> None:
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

    Inspection copies of a graph are not written here — they are artifacts,
    and go through :func:`paramem.utils.artifacts.on_fold_graph`.
    """
    from paramem.backup.encryption import write_infra_bytes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    payload = json.dumps(data, indent=2).encode("utf-8")
    write_infra_bytes(path, payload)
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

        ``key``, ``subject``, ``predicate``, ``object``, ``speaker_id``

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


def _add_keyed_edge(
    graph: nx.MultiDiGraph,
    subject: str,
    object_: str,
    *,
    indexed_key: str,
    predicate: str,
    speaker_id: str,
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
    """
    graph.add_edge(
        subject,
        object_,
        **{
            _IK_KEY_ATTR: indexed_key,
            "predicate": predicate,
            "speaker_id": speaker_id,
        },
    )


def build_tier_graph_from_store(store, tier: str) -> nx.MultiDiGraph:
    """Project the *tier* slice of a :class:`MemoryStore` into a fresh ``MultiDiGraph``.

    Enumerates active-only keys via ``store.tier_simhashes(tier, include_stale=False)``
    — the single accessor that makes the active-vs-known distinction explicit and
    impossible to forget.  Stale keys are excluded by construction (the
    ``include_stale=False`` call already filters them); the old belt-and-suspenders
    ``is_stale`` skip is removed because stale keys are never in the active-only map.

    Reads the matching entry from the store to add an edge
    ``(subject → object)`` with edge-data
    ``{ik_key, predicate, speaker_id}`` (the indexed-memory key is stored as
    ``"ik_key"`` to avoid the NetworkX ``"key"`` collision;
    :func:`iter_entries` maps it back to ``"key"`` for callers).  The store's
    entry cache carries content only (``subject``/``predicate``/``object``);
    ``speaker_id`` is attribution bookkeeping, so it is read from
    ``store.bookkeeping_for_key(indexed_key)`` instead — the single
    authority for who introduced a key.  A key with no bookkeeping record
    persists with ``speaker_id=""``.

    The caller is responsible for persisting the returned graph with
    :func:`save_memory_to_disk`.

    Args:
        store: A :class:`paramem.memory.store.MemoryStore`.
        tier: One of ``"episodic"``, ``"semantic"``, or ``"procedural"``.

    Returns:
        A new ``nx.MultiDiGraph`` with one edge per *active* key in the tier's
        SimHash registry.

    Raises:
        KeyError: When an active key is absent from the store's entry cache.
            A simhash/entry divergence on an active key is a data-integrity bug
            to surface, not paper over.
    """
    active_simhashes: dict[str, int] = store.tier_simhashes(tier, include_stale=False)
    graph = nx.MultiDiGraph()
    for indexed_key in active_simhashes:
        entry = store.get(indexed_key)
        if entry is None:
            raise KeyError(indexed_key)
        bookkeeping = store.bookkeeping_for_key(indexed_key)
        speaker_id = bookkeeping.get("speaker_id", "") if bookkeeping is not None else ""
        _add_keyed_edge(
            graph,
            entry["subject"],
            entry["object"],
            indexed_key=indexed_key,
            predicate=entry.get("predicate", ""),
            speaker_id=speaker_id,
        )
    return graph


def reap_tier_artifacts(tier_root: Path) -> list[Path]:
    """Remove one tier's on-disk artifacts, leaving the never-trained shape.

    The shape is derived from *tier_root* itself, never from a caller flag:

      * interim slot root (dir name starts with ``INTERIM_DIR_PREFIX``) — the
        whole directory is removed;
      * ``episodic`` root — every child EXCEPT ``interim_*`` containers is
        removed (those are separate interim tiers living under ``episodic/``
        — see :func:`~paramem.memory.interim_adapter.interim_dir_for_name`,
        the only place an interim tier container is ever created), then the
        root itself is removed too when the removal left it empty;
      * ``semantic``/``procedural`` root — every child, INCLUDING any
        ``interim_*``-named one, is removed. Interim tier CONTAINERS live
        only under ``episodic/``; an ``interim_<stamp>/`` dir under
        ``semantic/`` or ``procedural/`` is HF Trainer scratch
        (:meth:`paramem.training.consolidation.ConsolidationLoop._training_output_dir`
        builds one for any adapter's interim training scope), not a tier —
        sparing it here would leave scratch behind and the root would never
        become empty enough to ``rmdir``.

    In the main-tier branch, ``indexed_key_registry.json`` is condemned LAST
    by construction — it is the commit signal a boot sweep reads to decide
    whether a tier holds any keys, so a crash mid-reap must never leave the
    registry gone while a weight slot survives (see the ordering comment at
    the child-iteration site for the full rationale).

    Crash-safe deletion (rename-then-delete): every condemned root — the
    whole slot in the interim branch, each surviving child in the main-tier
    branch — is first ``os.rename``d into
    ``<adapter_dir>/.pending-delete/<name>`` (created on demand; a same-name
    collision means a prior crash left condemned debris there already, so it
    is removed first) and only then deleted. ``os.rename`` is atomic on
    POSIX/WSL2-ext4, so a crash before the rename leaves the live namespace
    untouched and a crash after it leaves the condemned tree already outside
    the live namespace — unambiguously gone from the boot sweep's point of
    view — rather than a registry-less corpse that looks identical to a torn
    *commit* (see :func:`commit_tier_slot`'s crash-semantics note, which is
    the shape this deliberately does NOT produce). :func:`resume_pending_reaps`
    finishes anything left stranded in ``.pending-delete/`` at boot.
    *adapter_dir* is derived from *tier_root*'s own shape (interim slot →
    grandparent; main tier → parent) rather than taken as a parameter, so
    every existing caller keeps working unchanged.

    ``INTERIM_DIR_PREFIX`` is imported lazily from
    :mod:`paramem.memory.interim_adapter` — a top-level import would create
    an import cycle now that :func:`paramem.memory.interim_adapter.unload_interim_adapters`
    imports this module's :func:`reap_tier_artifacts` at module scope (mirrors
    the existing lazy-import pattern used by :func:`commit_tier_slot` for
    :func:`~paramem.memory.interim_adapter.adapter_slot_root_for_name`).

    Args:
        tier_root: Either an interim slot directory
            (``<adapter_dir>/episodic/interim_<stamp>/``) or a main tier root
            (``<adapter_dir>/<episodic|semantic|procedural>/``).

    Returns:
        The removed paths, deepest-first and sorted, expressed at their
        ORIGINAL (pre-rename) locations — callers reason about the live
        namespace, not the transient tombstone shape (files and directories
        under the same parent are ordered so nothing is orphaned).  Empty
        list when *tier_root* does not exist. No-op-safe; idempotent — a
        second call on an already-reaped root returns ``[]``.
    """
    import os
    import shutil

    from paramem.memory.interim_adapter import INTERIM_DIR_PREFIX

    tier_root = Path(tier_root)
    if not tier_root.exists():
        return []

    is_interim_slot = tier_root.name.startswith(INTERIM_DIR_PREFIX)
    # Interim slot roots live at <adapter_dir>/episodic/interim_<stamp>/;
    # main tier roots live at <adapter_dir>/<tier>/ — both shapes are fixed
    # (see adapter_slot_root_for_name / interim_dir_for_name), so the
    # ancestor distance to adapter_dir is a reliable function of the shape
    # alone, never re-derived by any other means.
    adapter_dir = tier_root.parent.parent if is_interim_slot else tier_root.parent
    pending_dir = adapter_dir / _PENDING_DELETE_DIR_NAME

    def _condemn_and_remove(path: Path) -> list[Path]:
        if not path.exists():
            return []
        # Original (pre-rename) paths — walked before the rename so the
        # returned list always describes the live namespace as it existed
        # right before condemnation, never the transient tombstone shape.
        # A main tier root's direct children are a mix of plain files
        # (indexed_key_registry.json, graph.json) and directories (timestamped
        # weight slots); both go through the same rename-then-delete step —
        # a single file's unlink is already atomic, but routing it through
        # the tombstone too keeps one code path and one collision rule.
        if not path.is_dir():
            original_paths: list[Path] = [path]
        else:
            original_paths = [path]
            for dirpath, dirnames, filenames in os.walk(path):
                base = Path(dirpath)
                original_paths.extend(base / name for name in filenames)
                original_paths.extend(base / name for name in dirnames)

        pending_dir.mkdir(parents=True, exist_ok=True)
        tombstone = pending_dir / path.name
        if tombstone.exists():
            # Stale leftover from a prior crash — already-condemned debris.
            if tombstone.is_dir():
                shutil.rmtree(tombstone)
            else:
                tombstone.unlink()

        os.rename(path, tombstone)
        if tombstone.is_dir():
            shutil.rmtree(tombstone)
        else:
            tombstone.unlink()
        return original_paths

    removed: list[Path] = []
    if is_interim_slot:
        removed.extend(_condemn_and_remove(tier_root))
    else:
        # Interim tier containers (separate tiers, reaped by their own call)
        # live only under episodic/ — an interim_*-named child anywhere else
        # is training scratch, not a tier, and must not be spared.
        spare_interim_children = tier_root.name == "episodic"
        children = [
            child
            for child in sorted(tier_root.iterdir())
            if not (
                spare_interim_children
                and child.is_dir()
                and child.name.startswith(INTERIM_DIR_PREFIX)
            )
        ]
        # The registry file is condemned LAST, by construction — never by
        # sorted()'s incidental ASCII collation (which today happens to put
        # "indexed_key_registry.json" after digit-named slot dirs, but
        # states nothing about that ordering and nothing pins it). The
        # registry is the commit signal the boot sweep
        # (_sweep_keyless_tier_artifacts, paramem/server/app.py) reads to
        # decide whether a tier holds any keys, so a crash mid-reap must
        # always leave "registry present, some/all weight slots gone" —
        # never "registry gone, a weight slot still present", which nothing
        # else would revisit. A stable sort on "is this the registry"
        # preserves relative order among every other child.
        children.sort(key=lambda c: c.name == "indexed_key_registry.json")
        for child in children:
            removed.extend(_condemn_and_remove(child))
        if not any(tier_root.iterdir()):
            tier_root.rmdir()
            removed.append(tier_root)

    # Every condemned entry is deleted from the tombstone dir immediately
    # after its own rename (inside _condemn_and_remove), so a fully
    # successful reap leaves .pending-delete/ empty — clean it up rather
    # than leaving an empty dir behind on every reap.
    if pending_dir.exists() and not any(pending_dir.iterdir()):
        pending_dir.rmdir()

    return sorted(removed, key=lambda p: (-len(p.parts), str(p)))


def resume_pending_reaps(adapter_dir: Path) -> None:
    """Finish any :func:`reap_tier_artifacts` deletion interrupted mid-rmtree.

    :func:`reap_tier_artifacts` rename-condemns every root it deletes into
    ``<adapter_dir>/.pending-delete/<name>`` before deleting it there, so a
    crash between the rename and the delete leaves the condemned tree
    stranded under the tombstone dir — already out of the live namespace,
    but not yet actually gone from disk. Call this once at boot, before any
    live-namespace tier scan (mirrors the existing
    ``sweep_orphan_pending``-before-``find_live_slot`` ordering in
    :mod:`paramem.backup.atomic`), to finish those deletions and remove the
    now-empty tombstone dir.

    Best-effort: this is called from ``_load_model_into_state`` during boot,
    and every other failure mode its caller
    (``_sweep_keyless_tier_artifacts``, ``paramem/server/app.py``) can hit is
    already log-and-continue — an unguarded exception here would be the one
    way this sweep aborts the entire boot instead. A single entry that
    cannot be removed (permission error, still held open, etc.) is logged at
    ERROR and skipped rather than raised; it is retried on the next boot.
    The tombstone directory itself existing as a regular file (an
    unexpected shape, but not a reason to abort boot) is tolerated the same
    way — unlinked directly instead of raising ``NotADirectoryError`` out of
    ``iterdir()``.

    Idempotent — a missing or already-empty ``.pending-delete`` dir is a
    silent no-op. One scan root only: this does not walk into any other
    directory under *adapter_dir*.

    Args:
        adapter_dir: Adapter root (``config.adapter_dir``).

    Returns:
        ``None``.
    """
    import shutil

    adapter_dir = Path(adapter_dir)
    pending_dir = adapter_dir / _PENDING_DELETE_DIR_NAME
    if not pending_dir.exists():
        return

    if not pending_dir.is_dir():
        try:
            pending_dir.unlink()
        except OSError:
            logger.error(
                "resume_pending_reaps: %s is a file, not a directory, and could "
                "not be removed — left for next boot",
                pending_dir,
                exc_info=True,
            )
        return

    for entry in sorted(pending_dir.iterdir()):
        logger.warning(
            "resume_pending_reaps: resuming interrupted deletion of %s",
            entry,
        )
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError:
            logger.error(
                "resume_pending_reaps: failed to remove %s — left stranded, "
                "will retry on next boot",
                entry,
                exc_info=True,
            )

    try:
        pending_dir.rmdir()
    except OSError:
        # Non-empty because at least one entry above failed to be removed —
        # leave the tombstone dir itself for the next boot's resume rather
        # than raising out of this best-effort sweep.
        logger.error(
            "resume_pending_reaps: %s not empty after resume attempt — left for next boot",
            pending_dir,
        )


def erase_keys_from_graph_file(path: Path, keys: set[str]) -> int:
    """Remove every edge whose ``ik_key`` is in *keys* from the graph at *path*.

    Returns the number of edges removed; 0 (and no write) when the file does
    not exist or holds no matching edge.  Nodes left with degree 0 are
    dropped.  Read and write both go through the encryption-aware pair
    (:func:`load_memory_from_disk` / :func:`save_memory_to_disk`), so an
    age-wrapped graph stays age-wrapped and the write stays atomic.

    *keys* must be passed as a ``set`` — the membership test runs once per
    edge, so a list argument would make the pass O(edges x keys).

    Args:
        path: Tier ``graph.json`` path (main tier root or interim slot root).
        keys: Indexed-memory keys to erase.

    Returns:
        Number of edges removed.
    """
    path = Path(path)
    if not path.exists():
        return 0

    graph = load_memory_from_disk(path)
    to_remove = [
        (subject, object_, nx_edge_key)
        for subject, object_, nx_edge_key, data in graph.edges(keys=True, data=True)
        if data.get(_IK_KEY_ATTR) in keys
    ]
    if not to_remove:
        return 0

    for subject, object_, nx_edge_key in to_remove:
        graph.remove_edge(subject, object_, key=nx_edge_key)

    graph.remove_nodes_from(list(nx.isolates(graph)))

    save_memory_to_disk(graph, path)
    return len(to_remove)


def erase_keys_and_restamp_manifest(
    *,
    store,
    adapter_dir: Path,
    keys: list[str],
) -> dict[str, Path]:
    """Hard-erase *keys* from every tier that knows one, keeping registry,
    graph content, and (for a surviving tier) the weight-slot manifest in
    lockstep.

    Per affected tier, in order:

    1. Resolve the on-disk slot root
       (:func:`~paramem.memory.interim_adapter.adapter_slot_root_for_name`)
       and read the pre-erase registry hash
       (:func:`~paramem.adapters.manifest.tier_registry_sha256`) — both
       BEFORE ``store.discard_keys`` runs, so a malformed tier name aborts
       before any mutation and the hash reflects exactly what the last save
       wrote to disk (not a re-serialisation of the in-memory registry,
       which could diverge from the stamped manifest hash across a future
       payload-shape change).
    2. ``store.discard_keys(keys, mode="erase")`` — hard erase from every
       tier's ``KeyRegistry`` (active + stale + simhash) and bookkeeping in
       one call, across every tier at once.
    3. ``registry.save(...)`` (durable erase) then
       :func:`erase_keys_from_graph_file` (fact content) — registry first,
       content second, so a crash between the two never leaves an orphaned
       graph edge no reader can resolve.
    4. A surviving tier (still knows a key after the erase) gets its live
       weight slot's manifest re-stamped so
       :func:`~paramem.adapters.manifest.find_live_slot` rebinds it on
       restart. Gated by a venue check
       (:func:`~paramem.adapters.manifest.count_slot_candidates` — a weight
       slot can only exist in the train venue; zero candidates means there
       is no slot to re-stamp, not that one is orphaned) and an empty-hash
       guard (an unreadable/absent pre-erase registry must never bind a
       stray ``""``-stamped slot). A tier emptied by the erase is not
       re-stamped — it is returned in the result for the caller to reap.

    A no-op — returns ``{}``, calls nothing — when *keys* is empty.

    Args:
        store: A :class:`~paramem.memory.store.MemoryStore` (duck-typed:
            only ``tiers_with_registry``, ``registry``, and ``discard_keys``
            are used). Production caller passes
            :attr:`~paramem.training.consolidation.ConsolidationLoop.store`.
        adapter_dir: Adapter store root the tier slot roots are resolved
            under. Production caller passes ``config.adapter_dir``.
        keys: Indexed-memory keys to erase. Production caller passes the
            keys resolved for the target speaker via
            ``store.iter_bookkeeping()``.

    Returns:
        Tier name -> resolved slot-root path, one entry per tier the erase
        reduced to zero known keys. Empty when every affected tier still
        knows at least one key (or *keys* was empty).

    Raises:
        ValueError: A tier name under *adapter_dir* does not parse as a
            valid slot path (propagated from
            :func:`~paramem.memory.interim_adapter.adapter_slot_root_for_name`),
            raised before ``discard_keys`` runs so no mutation has happened
            yet.
    """
    if not keys:
        return {}

    import hashlib as _hashlib
    from dataclasses import replace as _replace

    from paramem.adapters.manifest import (
        count_slot_candidates,
        find_live_slot,
        read_manifest,
        tier_registry_sha256,
        write_manifest,
    )
    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    keys_set = set(keys)

    tier_root: dict[str, Path] = {}
    tier_pre_sha: dict[str, str] = {}
    for tier_name in store.tiers_with_registry():
        registry = store.registry(tier_name)
        if any(registry.knows(k) for k in keys):
            root = adapter_slot_root_for_name(adapter_dir, tier_name)
            tier_root[tier_name] = root
            tier_pre_sha[tier_name] = tier_registry_sha256(root)

    store.discard_keys(keys, mode="erase")

    emptied_tiers: dict[str, Path] = {}
    for tier_name, root in tier_root.items():
        registry = store.registry(tier_name)
        registry.save(root / "indexed_key_registry.json")
        erase_keys_from_graph_file(root / "graph.json", keys_set)

        if not registry.list_known():
            logger.info(
                "erase_keys_and_restamp_manifest: tier %s reduced to zero known keys",
                tier_name,
            )
            emptied_tiers[tier_name] = root
            continue

        pre_sha = tier_pre_sha[tier_name]
        has_weight_slot = count_slot_candidates(root) > 0
        if not has_weight_slot:
            logger.debug(
                "erase_keys_and_restamp_manifest: tier %s has no on-disk weight slot — "
                "skipping manifest re-stamp (simulate venue or never-trained tier)",
                tier_name,
            )
        elif pre_sha == "":
            logger.warning(
                "erase_keys_and_restamp_manifest: tier %s had no readable pre-erase "
                "registry on disk (empty hash) — skipping manifest re-stamp to avoid "
                'binding a stray ""-stamped slot; recover via consolidation fold or '
                "registry restore",
                tier_name,
            )
        else:
            new_hash = _hashlib.sha256(registry.save_bytes()).hexdigest()
            slot = find_live_slot(root, pre_sha)
            if slot is None:
                logger.error(
                    "erase_keys_and_restamp_manifest: tier %s live slot for pre-erase "
                    "hash %s… not found — slot already orphaned; recover via "
                    "consolidation fold or registry restore",
                    tier_name,
                    pre_sha[:12],
                )
            else:
                write_manifest(
                    slot,
                    _replace(
                        read_manifest(slot), registry_sha256=new_hash, key_count=len(registry)
                    ),
                )

        logger.info(
            "erase_keys_and_restamp_manifest: removed key(s) from KeyRegistry tier %s",
            tier_name,
        )

    return emptied_tiers


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


def commit_tier_slot(
    *,
    loop: "ConsolidationLoop",
    tier: str,
    adapter_name: str,
    stamp: str,
    mode: "Literal['simulate', 'train']",
    all_keyed: "list[dict]",
    output_dir: Path,
    verify: "Callable[[Path], None] | None" = None,
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

       - **Train mode**: :func:`paramem.models.loader.save_adapter`
         writes the PEFT adapter weights into a timestamped slot dir under the
         root.  The manifest (from step 3) is embedded as ``meta.json``.
         If *verify* is not ``None``, it is called with the slot path
         immediately after the weight write and before the registry flush
         (step 6).  A raise from *verify* propagates as-is; the ``finally``
         orphan-cleanup fires because ``_registry_flushed`` is still ``False``
         at that point, removing the half-committed slot.
       - **Simulate mode**: builds a ``MultiDiGraph`` from *all_keyed* and
         writes it as ``<slot_root>/graph.json`` via :func:`save_memory_to_disk`
         (encrypted/plaintext depending on daily-key state).  No PEFT weights
         are written.  *verify* is not called in simulate mode.

    6. ``loop.write_key_metadata()`` — durably persist per-key bookkeeping
       (speaker_id, relation_type, reinforcement_count, ...) to
       ``key_metadata.json``, ordered before the registry flush (step 7) so a
       newly-minted key's bookkeeping row is on disk before the registry
       write that makes the key discoverable on the next boot.

    7. Flush the exact registry bytes from step 1 to
       ``<slot_root>/indexed_key_registry.json`` as the commit signal (both
       modes) via :meth:`paramem.training.key_registry.KeyRegistry.save_from_bytes`.
       This is the last write — its presence on disk signals that all preceding
       files are complete.

    Crash semantics: a kill after step 5 but before step 7 leaves the slot
    present without the registry file.  Nothing at boot deletes this shape:
    the keyless-tier sweep
    (:func:`paramem.server.app._sweep_keyless_tier_artifacts`) only acts on
    a tier whose ``indexed_key_registry.json`` exists and affirmatively
    reads zero known keys, so a registry-absent tier is skipped outright —
    not this sweep's business, per its own docstring.  The mount loop in
    :func:`paramem.server.app._mount_adapters_from_slots` then finds no
    matching slot for the interim tier's (degraded) hash and logs the
    payload-bearing-but-unmatched slot as an ERROR, leaving it on disk. The
    partial slot survives across restarts until an operator or a later fold
    reuses or clears it.

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
        verify: Optional callable invoked with the written slot ``Path``
            after weight write and **before** the registry flush (train mode
            only).  A raise propagates unchanged; the ``finally`` orphan-cleanup
            removes the slot because ``_registry_flushed`` is still ``False``.
            Pass ``None`` (default) to skip verification (simulate mode or
            callers that do not need disk-integrity gating).

    Returns:
        ``None``.  Raises on I/O failure or verify failure.

    Raises:
        RuntimeError: When ``loop.store.replay_enabled`` is ``False`` (no
            registry to commit).
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
    # registry() always returns a KeyRegistry (setdefault auto-creates on first
    # access; never returns None).  The replay_enabled gate above is the only
    # guard; no None check is needed here.
    tier_reg = loop.store.registry(adapter_name)

    # Empty commit is a real production path: len(tier_reg) == 0 is permitted
    # and produces an empty but valid registry file.  No zero-key guard is added.

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
                key_count=len(tier_reg),
                base_model_hash_cache=fingerprint_cache,
                registry_sha256_override=registry_sha256,
                window_stamp=stamp,
                adapter_root=output_dir,
            )

            # Use save_adapter (thin forwarder to atomic_save_adapter) so tests can
            # patch paramem.models.loader.save_adapter without reaching atomic internals.
            # save_adapter now returns the timestamped slot Path so the verify
            # callback receives the exact directory that was written.
            from paramem.models.loader import save_adapter as _save_adapter

            slot = _save_adapter(loop.model, slot_root, adapter_name, manifest=manifest)
            logger.debug(
                "commit_tier_slot: adapter weights saved for %s (tier=%s)",
                adapter_name,
                tier,
            )
            # Disk-integrity gate: runs BEFORE the registry flush (step 6) so
            # that a failed verify fires while _registry_flushed is still False.
            # The finally orphan-cleanup then removes the half-committed slot.
            # GPU recall in 'simulate' mode is a no-op path — verify is None there.
            if verify is not None:
                verify(slot)
        else:
            # Simulate mode: build graph from all_keyed and write graph.json.
            # When all_keyed is empty, re-project from the canonical store so
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
                    )
            else:
                # Caller passed [] — re-project from the canonical store.
                graph = build_tier_graph_from_store(loop.store, adapter_name)
            graph_path = slot_root / "graph.json"
            save_memory_to_disk(graph, graph_path)
            logger.debug(
                "commit_tier_slot: graph.json written for %s (tier=%s, %d edges)",
                adapter_name,
                tier,
                graph.number_of_edges(),
            )

        # --- Step 6: Bookkeeping — durable BEFORE the registry flush (step 7) ---
        # The registry flush below is the commit signal that makes this
        # slot's keys discoverable on the next boot, so their bookkeeping
        # rows in key_metadata.json must already be on disk by the time that
        # signal lands.  A raise here fires before _registry_flushed is set,
        # so the finally block below removes the (still-orphan) slot dir.
        loop.write_key_metadata()

        # --- Step 7: Registry flush — commit signal (both modes, LAST write) ---
        # The registry now carries the unified simhash map (active∪stale) in its
        # "simhash" key, so a separate simhash_registry.json is no longer written.
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
