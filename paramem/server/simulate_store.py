"""Simulate-mode graph persistence for indexed-key memory.

Simulate mode uses a NetworkX ``MultiDiGraph`` on disk instead of adapter
weights as its storage medium.  The per-tier layout mirrors the canonical
``<tier>/keyed_pairs.json`` layout 1:1 — only the persistence bytes change.

On-disk format: ``nx.node_link_data(graph)`` → JSON → optional age-encrypt →
atomic write.  The wire format is identical to the cumulative knowledge graph
written by :mod:`paramem.graph.merger` so existing encryption infrastructure
is reused without modification.

Public API
----------
- :func:`save_simulate_graph` — atomic encrypted write.
- :func:`load_simulate_graph` — decryption-aware read; empty graph on miss.
- :func:`iter_quads` — yield quad dicts for every edge carrying an indexed key.
- :func:`quad_by_key` — look up a single key; ``None`` on miss.
- :func:`keys_for_entity` — edge keys where lower(subject) or lower(object)
  matches the query.
- :func:`keys_for_speaker` — edge keys where ``speaker_id`` matches.
- :func:`build_tier_graph_from_loop` — project in-memory loop state to a
  fresh ``MultiDiGraph`` for a given tier.

Internal edge attribute naming
-------------------------------
NetworkX's ``node_link_data`` serialisation format uses ``"key"`` as the
reserved field name for the multigraph edge-key integer in the JSON output.
To avoid collision, the indexed-memory key string is stored as the internal
edge-data attribute ``"ik_key"``.  All public API functions
(:func:`iter_quads`, :func:`quad_by_key`, :func:`keys_for_entity`,
:func:`keys_for_speaker`) map ``"ik_key"`` to ``"key"`` in the dict they
return so callers see the canonical ``KEYED_PAIR_FIELDS_QUAD`` shape.

The six-field public schema matches ``KEYED_PAIR_FIELDS_QUAD``:

    ``key``, ``subject``, ``predicate``, ``object``,
    ``speaker_id``, ``first_seen_cycle``

where ``subject`` and ``object`` are the graph node endpoints and the
remaining four fields come from edge attributes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

# Internal attribute name for the indexed-memory key on graph edges.
# Must not be "key" — that is the NetworkX reserved multigraph edge-key
# field in node_link_data JSON output and would be lost on round-trip.
_IK_KEY_ATTR = "ik_key"


def save_simulate_graph(graph: nx.MultiDiGraph, path: Path, *, encrypted: bool = True) -> None:
    """Atomic encrypted write of *graph* to *path*.

    Serialises via ``nx.node_link_data`` → JSON → bytes, then delegates to
    the infrastructure envelope so the result is age-encrypted when a daily
    identity is loaded, and plaintext otherwise.  The write is atomic:
    ``<path>.tmp`` is written, fsynced, and renamed in a single step so a
    crash leaves no partial file.

    The indexed-memory key is stored as the ``"ik_key"`` edge attribute to
    avoid collision with the NetworkX-reserved ``"key"`` serialisation field.
    :func:`iter_quads` and friends map ``"ik_key"`` back to ``"key"`` in the
    public-facing quad dicts.

    Args:
        graph: The ``MultiDiGraph`` to persist.
        path: Destination path (e.g. ``simulate_dir/episodic/graph.json``).
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
    logger.debug("Simulate graph saved to %s (%d edges)", path, graph.number_of_edges())


def load_simulate_graph(path: Path) -> nx.MultiDiGraph:
    """Decryption-aware load of a simulate graph from *path*.

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
        logger.debug("No simulate graph at %s — returning empty MultiDiGraph", path)
        return nx.MultiDiGraph()

    raw = read_maybe_encrypted(path)
    data = json.loads(raw.decode("utf-8"))
    graph = nx.node_link_graph(data, multigraph=True, directed=True)
    logger.debug(
        "Simulate graph loaded from %s: %d nodes, %d edges",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def iter_quads(graph: nx.MultiDiGraph) -> Iterator[dict]:
    """Yield quad dicts for every edge in *graph* that carries an indexed-memory key.

    Edges without the ``"ik_key"`` internal attribute (e.g. cumulative-graph
    edges that pre-date key assignment) are silently skipped so this function
    is safe to call on mixed-content graphs.

    Each yielded dict has exactly the six ``KEYED_PAIR_FIELDS_QUAD`` fields:

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
        Dicts with the six ``KEYED_PAIR_FIELDS_QUAD`` fields.
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


def quad_by_key(graph: nx.MultiDiGraph, key: str) -> dict | None:
    """Return the quad dict for *key*, or ``None`` when the key is absent.

    Linear scan over all edges (graphs are per-tier, expected to contain at
    most a few hundred edges).  Returns the first matching edge without
    examining further edges, so the function is ``O(n)`` worst-case.

    Args:
        graph: Source ``MultiDiGraph``.
        key: The indexed-memory key to look up (e.g. ``"graph1"``).

    Returns:
        Quad dict with six ``KEYED_PAIR_FIELDS_QUAD`` fields on hit;
        ``None`` on miss.
    """
    for quad in iter_quads(graph):
        if quad["key"] == key:
            return quad
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
    for quad in iter_quads(graph):
        if quad["subject"].lower() == entity_lower or quad["object"].lower() == entity_lower:
            matched.add(quad["key"])
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
    for quad in iter_quads(graph):
        if quad["speaker_id"] == speaker_id:
            matched.add(quad["key"])
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

    :func:`iter_quads` and friends map ``"ik_key"`` back to ``"key"`` so
    callers see the canonical ``KEYED_PAIR_FIELDS_QUAD`` shape.

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


def build_tier_graph_from_loop(loop, tier: str) -> nx.MultiDiGraph:
    """Project in-memory loop state for *tier* into a fresh ``MultiDiGraph``.

    Iterates every key in ``loop.<tier>_simhash`` and reads the matching
    entry from ``loop.indexed_key_cache`` to add an edge
    ``(subject → object)`` with edge-data
    ``{ik_key, predicate, speaker_id, first_seen_cycle}`` (the indexed-memory
    key is stored as ``"ik_key"`` to avoid the NetworkX ``"key"`` collision;
    :func:`iter_quads` maps it back to ``"key"`` for callers).

    The caller is responsible for persisting the returned graph with
    :func:`save_simulate_graph`.

    Args:
        loop: A ``ConsolidationLoop`` instance (or any object that exposes
            ``<tier>_simhash: dict[str, int]`` and
            ``indexed_key_cache: dict[str, dict]``).
        tier: One of ``"episodic"``, ``"semantic"``, or ``"procedural"``.

    Returns:
        A new ``nx.MultiDiGraph`` with one edge per key in the tier's
        SimHash registry.

    Raises:
        KeyError: When a key present in ``<tier>_simhash`` is absent from
            ``loop.indexed_key_cache``.  This is *stricter* than the
            current ``_write_keyed_pairs`` writer at
            :mod:`paramem.server.consolidation` (which silently filters
            missing entries via ``if k in indexed_key_cache``); the
            simulate-mode graph treats a simhash/cache divergence as a
            data-integrity bug to surface, not paper over.  When the kp
            writer is removed in the train-mode kp sunset, the contracts
            will converge on this fail-fast behavior.
    """
    simhash_registry: dict[str, int] = getattr(loop, f"{tier}_simhash")
    graph = nx.MultiDiGraph()
    for indexed_key in simhash_registry:
        # Fail-fast: KeyError surfaces here if the cache is incomplete.
        entry = loop.indexed_key_cache[indexed_key]
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
