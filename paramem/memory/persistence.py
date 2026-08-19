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
- :func:`erase_keys_and_restamp_manifest` — a file surgeon: reads every
  tier's persisted registry directly off disk, stale-marks a key set in
  every tier that holds one ACTIVE, and (for a bound slot) restamps its
  manifest, in one atomic-ordered sequence. Never refuses — every affected tier
  reported in the return value has its mutation landed and its rebind
  attempted; a tier left unbound is reported there, not raised, while a
  tier whose own registry write failed is never reported (it still
  propagates uncaught). No
  :class:`~paramem.memory.store.MemoryStore` or model is touched or
  required — shared by every out-of-fold registry-repair caller
  (``POST /speaker/forget``, ``POST /debug/erase-keys`` today).
- :func:`plan_restamp` — THE pure-read precondition-and-target resolver for
  a no-retrain registry commit: no slot and no active key (legal), no slot
  with an active key, no readable pre-write hash, or no slot matching
  either digest (all three refuse), otherwise a slot to rebind. Composed by
  both :func:`restamp_tier_manifest` (the write) and
  :func:`assert_publish_preconditions` (the publish preflight) — the one
  planner, never re-derived a second way.
- :func:`restamp_tier_manifest` — the no-retrain commit primitive
  ``erase_keys_and_restamp_manifest`` calls per tier: plans via
  :func:`plan_restamp`, writes a registry payload to disk, then — when the
  plan named a slot — rebinds that slot's manifest to the new hash. Always
  lands the registry; never refuses — no precondition raises before the
  mutation. THE classification site for a no-retrain commit's two I/O
  failure shapes: a registry-write ``OSError`` still propagates uncaught
  (nothing landed), while a rebind-phase ``OSError``/``ManifestError`` is
  caught right here and returned as :data:`REBIND_FAILED` — never raised —
  because this is the only place that knows the registry write already
  landed before the rebind pair could fail. The one place any caller — the
  erase door out-of-fold, or the fold's publish for a tier it did not
  retrain — commits a registry mutation without retraining.
- :func:`assert_publish_preconditions` — the automated publish preflight:
  validates every bundle member's publish preconditions (pure reads only)
  before :func:`publish_tier_registry`'s first durable write, and raises
  :class:`TierWriteRefused` naming every failing member so the whole bundle
  refuses with zero bytes written. The operator erase doors never call
  this — they have no refusal concept.
- :func:`commit_tier_slot` — the one per-tier commit primitive (interim slots,
  main tiers, and the trial tree all route through it): writes one payload
  into a fresh timestamped slot through the shared slot envelope
  (:func:`~paramem.adapters.slot.write_slot`), with the tier-root registry
  written last as the commit signal; mode-switches between adapter-weight
  venue (train, with a debug shadow write) and graph-JSON venue (simulate) —
  both venues get post-commit slot pruning.
- :func:`reap_tier_artifacts` — remove one tier's on-disk artifacts, shape derived
  from the tier root itself (interim slot vs. main tier). Rename-condemns
  each removed root into ``.pending-delete/`` before deleting it there, so a
  crash mid-delete leaves the corpse out of the live namespace instead of
  half-deleted in place.
- :func:`resume_pending_reaps` — boot-time sweep that finishes any deletion
  :func:`reap_tier_artifacts` left stranded under ``.pending-delete/``.
- :func:`write_tier_slot` — the write act of a two-phase training event's
  per-tier build -> gate -> write driver: writes one
  ``TierIncrement``'s payload (weight slot from the staging adapter, or the
  projected ``graph.json`` in the simulate venue) and returns the path
  written.  Writes nothing the live store reads.
- :func:`publish_tier_registry` — the commit signal of that same driver:
  writes the tier's rows then its registry payload, verbatim from the
  increment's bytes, either by flushing straight to a freshly written slot's
  paths or by routing through :func:`restamp_tier_manifest` for a member
  that written no payload. Always called AFTER
  :func:`assert_publish_preconditions` has already validated the whole
  bundle; its own rows-only-arm raise is a structural guard only.
- :func:`prune_old_slots` — module-level retention pass with bound-slot
  immunity, used by the go-live sequencer and the donor build.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    from paramem.memory.increment import TierIncrement, TierWriteContext
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.training.key_registry import KeyRegistry

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
        path: Destination path — the caller's own choice; production callers
            always pass a bound slot's payload path (e.g.
            ``adapter_dir/episodic/<ts>/graph.json``), never a tier-root
            path. Parent directory is created if absent.

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

    Enumerates active-only keys via ``store.tier_simhashes(tier)`` — the tier's
    one fingerprint map, active keys only, so no withheld id is in it by
    construction.

    Reads the matching entry from the store to add an edge
    ``(subject → object)`` with edge-data
    ``{ik_key, predicate, speaker_id}`` (the indexed-memory key is stored as
    ``"ik_key"`` to avoid the NetworkX ``"key"`` collision;
    :func:`iter_entries` maps it back to ``"key"`` for callers).  The store's
    entry cache carries content only (``subject``/``predicate``/``object``);
    ``speaker_id`` is attribution bookkeeping, so it is read from
    ``store.bookkeeping_for_key(indexed_key)`` instead — the single
    authority for who introduced a key.  Every active key already carries a
    bookkeeping row (the every-known-key-has-a-row invariant), so the row is
    read directly with no ``None`` fallback.

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
            A simhash/entry divergence on an active key is a data-integrity
            bug to surface, not paper over.
        TypeError: When an active key has no bookkeeping row — the same
            data-integrity bug on the bookkeeping side.
    """
    active_simhashes: dict[str, int] = store.tier_simhashes(tier)
    graph = nx.MultiDiGraph()
    for indexed_key in active_simhashes:
        entry = store.get(indexed_key)
        if entry is None:
            raise KeyError(indexed_key)
        speaker_id = store.bookkeeping_for_key(indexed_key)["speaker_id"]
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
    by construction — it is the commit signal a boot sweep
    (``_sweep_keyless_tier_artifacts``, ``paramem/server/app.py``) reads to
    decide whether a tier holds any keys, so a crash mid-reap must never
    leave the registry gone while a weight slot survives (see the ordering
    comment at the child-iteration site for the full rationale). That same
    boot sweep also detects the *other* direction — a registry that already
    reads zero known keys beside a slot whose binding does not independently
    corroborate that emptiness (a stale hash, a disagreeing ``key_count``) —
    but always preserves it as corruption rather than reaping it: the
    operator erase doors stale-mark rather than empty a tier, so there is no
    legitimate producer of a genuinely-emptied, unverified tier left to
    authorise a self-heal against.

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


# Status vocabulary for a no-retrain registry commit. The first five are
# shared by :func:`plan_restamp` and :func:`restamp_tier_manifest` — every
# no-retrain registry commit (the automated publish's rows-only members and
# both operator erase doors) reports one of them. :data:`REBIND_FAILED` is
# minted only by :func:`restamp_tier_manifest` itself, when the rebind pair
# that follows a RESTAMPED plan (the manifest re-read or its replacement
# write) raises ``OSError`` or :class:`~paramem.adapters.manifest.ManifestError`
# — that tier's registry mutation already landed (the write precedes the
# rebind pair unconditionally), only the slot rebind did not, and the
# failure is recorded as this outcome rather than propagated. Every caller
# of :func:`restamp_tier_manifest` — the erase door's mutation loop and
# :func:`publish_tier_registry` alike — receives it the same way, as a
# returned status, never an exception.
RESTAMPED: Final[str] = "restamped"
NOTHING_TO_BIND: Final[str] = "nothing_to_bind"
KEYS_WITHOUT_SLOT: Final[str] = "keys_without_slot"
NO_PRE_WRITE_HASH: Final[str] = "no_pre_write_hash"
SLOT_ORPHANED: Final[str] = "slot_orphaned"
REBIND_FAILED: Final[str] = "rebind_failed"


@dataclass(frozen=True)
class RestampPlan:
    """The precondition-and-target resolution :func:`plan_restamp` returns.

    Attributes:
        status: One of :data:`RESTAMPED`, :data:`NOTHING_TO_BIND`,
            :data:`KEYS_WITHOUT_SLOT`, :data:`NO_PRE_WRITE_HASH`, or
            :data:`SLOT_ORPHANED`.
        slot: The slot :func:`restamp_tier_manifest` should rebind, when
            *status* is :data:`RESTAMPED`; ``None`` for every other status.
    """

    status: str
    slot: Path | None


def plan_restamp(
    tier_root: Path, *, registry: "KeyRegistry", payload: bytes, pre_sha: str
) -> RestampPlan:
    """THE precondition-and-target resolver for a no-retrain registry commit.

    A pure read — no write of any kind, no side effect. Both
    :func:`restamp_tier_manifest` (which re-reads it at the moment of the
    write) and the publish preflight (:func:`assert_publish_preconditions`,
    for a rows-only bundle member) call this as the ONE planner; neither
    resolves the precondition set a second, independent way.

    Five rules, checked in order:

    1. No slot candidate anywhere under *tier_root*
       (:func:`~paramem.adapters.manifest.count_slot_candidates` == 0) and
       *registry* carries no active key (``len(registry) == 0``) — legal,
       nothing to bind. Returns :data:`NOTHING_TO_BIND`.
    2. No slot candidate, but *registry* carries at least one active key —
       a tier that should have a slot has none. Returns
       :data:`KEYS_WITHOUT_SLOT`.
    3. *pre_sha* is ``""`` (no readable registry existed on disk before this
       call) — binding a stray ``""``-stamped slot would be wrong. Returns
       :data:`NO_PRE_WRITE_HASH`.
    4. No on-disk slot's manifest matches *pre_sha* NOR
       ``sha256(payload)`` — the slot is already orphaned (e.g. a prior
       crash) and this call must not adopt it. Returns :data:`SLOT_ORPHANED`.
    5. Otherwise legal — a slot was found. Returns :data:`RESTAMPED` naming
       that slot.

    Slot resolution (rules 4/5) is TWO-DIGEST and that is load-bearing:
    ``find_live_slot(tier_root, pre_sha) or find_live_slot(tier_root,
    sha256(payload))``. A crash between a PRIOR call's manifest write and
    its caller's own follow-up can leave the manifest already carrying the
    new digest; a *pre_sha*-only lookup would then plan
    :data:`SLOT_ORPHANED` with no target, and a re-issue of identical bytes
    must resolve the same slot and plan :data:`RESTAMPED` instead. The
    second lookup is equally correct for a fresh call: a slot already
    stamped to the digest of the registry now being written already
    carries this call's own write.

    Args:
        tier_root: Resolved tier slot root (main tier or interim slot) —
            the same shape :func:`~paramem.adapters.manifest.tier_registry_sha256`
            and :func:`~paramem.adapters.manifest.find_live_slot` expect.
        registry: The tier's :class:`~paramem.training.key_registry.KeyRegistry`,
            already reflecting the caller's mutation. Used here only for its
            ``len()`` (rules 1/2's active-key count).
        payload: The exact bytes the caller is about to write (or just
            wrote) — ``registry.save_bytes()``. Hashed for rules 4/5's
            second lookup digest; never read for its content otherwise.
        pre_sha: The tier's registry hash from BEFORE the caller's
            mutation — the caller's own pre-write read
            (:func:`~paramem.adapters.manifest.tier_registry_sha256`).

    Returns:
        A :class:`RestampPlan` naming the outcome and, only for
        :data:`RESTAMPED`, the slot to rebind.
    """
    import hashlib as _hashlib

    from paramem.adapters.manifest import count_slot_candidates, find_live_slot

    if count_slot_candidates(tier_root) == 0:
        if len(registry) == 0:
            return RestampPlan(status=NOTHING_TO_BIND, slot=None)
        return RestampPlan(status=KEYS_WITHOUT_SLOT, slot=None)

    if pre_sha == "":
        return RestampPlan(status=NO_PRE_WRITE_HASH, slot=None)

    new_hash = _hashlib.sha256(payload).hexdigest()
    slot = find_live_slot(tier_root, pre_sha) or find_live_slot(tier_root, new_hash)
    if slot is None:
        return RestampPlan(status=SLOT_ORPHANED, slot=None)

    return RestampPlan(status=RESTAMPED, slot=slot)


@dataclass(frozen=True)
class RestampResult:
    """Outcome of one :func:`restamp_tier_manifest` call, or of one tier's
    entry in :func:`erase_keys_and_restamp_manifest`'s per-tier report.

    Attributes:
        status: One of :data:`RESTAMPED`, :data:`NOTHING_TO_BIND`,
            :data:`KEYS_WITHOUT_SLOT`, :data:`NO_PRE_WRITE_HASH`, or
            :data:`SLOT_ORPHANED` — :func:`plan_restamp`'s own vocabulary,
            carried straight through by :func:`restamp_tier_manifest`; or
            :data:`REBIND_FAILED`, minted only by
            :func:`restamp_tier_manifest` itself, when the rebind pair
            following a RESTAMPED plan raises ``OSError`` or
            :class:`~paramem.adapters.manifest.ManifestError`. Every
            status this class ever carries means the registry mutation
            already landed — the one status that would mean otherwise (a
            registry-write ``OSError``) is never wrapped into a
            :class:`RestampResult`; it propagates as a raised exception
            instead.
        slot: The re-stamped slot directory when ``status == RESTAMPED``;
            ``None`` for every other status.
        message: ``None`` for every status :func:`restamp_tier_manifest`
            itself returns; the caught exception's ``str()`` when
            ``status == REBIND_FAILED``.
    """

    status: str
    slot: Path | None
    message: str | None = None


class TierWriteRefused(RuntimeError):
    """One or more publish-bundle members failed their publish preconditions.

    Raised by :func:`assert_publish_preconditions` — the FIRST statement of
    the automated publish sequence
    (:func:`~paramem.training.go_live.publish_bundle`), before its first
    durable write — naming every failing member at once (the precondition
    loop over the whole bundle completes before this raises), so a refusal
    leaves ZERO bytes written for the whole bundle: the stage-ledger record
    stays pending, the shadow tree is untouched, and every contributing
    session stays pending.

    Also raised, as a structural guard only, by
    :func:`publish_tier_registry`'s rows-only arm for a caller that reaches
    it without running the preflight first. Dead on the automated publish
    path itself — the preflight already checked every member — so a live
    raise there means a concurrent mutation changed the tier underneath
    this event; loud after a landed write is the correct severity for that
    case.

    The operator erase doors (:func:`erase_keys_and_restamp_manifest`)
    never raise this: they have no refusal concept — every mutation lands
    and every rebind is attempted, with the per-tier outcome reported
    instead.

    Attributes:
        refusals: ``{tier_name: reason}`` for every failing member — one of
            :func:`plan_restamp`'s statuses for a rows-only member, or one
            of :func:`_written_member_refusal`'s reasons for a written member.
    """

    def __init__(self, refusals: dict[str, str]):
        self.refusals = dict(refusals)
        detail = ", ".join(f"{tier}={reason}" for tier, reason in self.refusals.items())
        super().__init__(f"publish refused for tier(s): {detail}")


def restamp_tier_manifest(
    tier_root: Path, *, registry: "KeyRegistry", payload: bytes, pre_sha: str
) -> RestampResult:
    """THE no-retrain commit for a tier whose payload did not change.

    Plans via :func:`plan_restamp` (a pure read), writes *payload* to
    ``tier_root/indexed_key_registry.json`` verbatim, then — only when the
    plan named a slot (``status == RESTAMPED``) — rebinds that slot's
    manifest to the freshly written registry's digest:
    ``write_manifest(slot, replace(read_manifest(slot),
    registry_sha256=new_hash, key_count=len(registry)))``. Always lands the
    registry; never refuses — there is no precondition that raises before
    the mutation, so the registry write is unconditional.

    Two different failure shapes after that point, deliberately handled
    differently:

    * The registry write itself (``registry.save_from_bytes``) can raise
      ``OSError``. Nothing has landed yet when this happens — no plan
      status has been acted on — so this propagates uncaught from every
      call site. Reporting anything here as a landed outcome would be
      false.
    * The rebind pair that follows a :data:`RESTAMPED` plan — the
      ``read_manifest(plan.slot)`` re-read (can raise
      :class:`~paramem.adapters.manifest.ManifestError`) and the
      ``write_manifest`` call (can raise ``OSError``) — is wrapped in
      ``except (OSError, ManifestError)`` right here, because this is the
      only place that knows the registry write already landed before
      either of these could fail. A caught failure here is returned as
      :data:`REBIND_FAILED` (the exception's message in
      ``RestampResult.message``) rather than raised: every caller
      receiving :data:`REBIND_FAILED` from this function can therefore
      rely on "the registry mutation landed" as a structural guarantee,
      never a claim it has to re-derive from which phase raised.

    Used by every caller that mutates a tier's registry without
    retraining: an operator door (:func:`erase_keys_and_restamp_manifest`)
    or :func:`publish_tier_registry`, for a bundle member that written no
    payload this event.

    This function performs NO serialization of its own: *payload* is the
    exact bytes to write — ``registry.save_bytes()``, called once by the
    caller — and it writes those bytes and stamps ``sha256(payload)``, so
    the on-disk file and the re-stamped manifest can never disagree with
    each other. *pre_sha* is likewise always the caller's own pre-mutation
    read (:func:`~paramem.adapters.manifest.tier_registry_sha256`, taken
    BEFORE the caller's mutation) — there is no read-inside fallback; a
    caller that mutates nothing before calling this function still reads
    its own pre-write hash first.

    Ordering is plan, then registry, then (conditionally) manifest,
    deliberately: the plan is a pure read that touches nothing, so a crash
    can only ever land between the registry write and the manifest write —
    leaving :func:`~paramem.adapters.manifest.find_live_slot` with no match
    for the new hash (surfaced, recoverable via a consolidation fold or
    registry restore) rather than a manifest bound to the wrong bytes.
    ``payload.sha256`` is never touched by a restamp: the member's payload
    did not change — only its ``registry_sha256`` and ``key_count`` did.

    Every caller sees the same contract: a registry-write ``OSError``
    still propagates uncaught (nothing landed), while a rebind-phase
    ``OSError``/``ManifestError`` never reaches a caller as an exception —
    it always comes back as :data:`REBIND_FAILED`.
    :func:`erase_keys_and_restamp_manifest`'s mutation loop relies on this
    to record a per-tier outcome without catching anything itself;
    :func:`publish_tier_registry` relies on it to fold
    :data:`REBIND_FAILED` into its own existing "not a clean publish"
    check (``result.status not in (RESTAMPED, NOTHING_TO_BIND)``) rather
    than needing a dedicated except clause.

    There is deliberately NO emptied-tier rule here: whether *registry* has
    been reduced to zero known keys is the caller's decision. A tier
    restamped here with ``key_count=0`` is strictly safer than leaving it
    unstamped: a crash immediately afterward leaves a self-consistent empty
    tier rather than an ambiguous one.

    Args:
        tier_root: Resolved tier slot root (main tier or interim slot) —
            the same shape :func:`plan_restamp` expects.
        registry: The tier's :class:`~paramem.training.key_registry.KeyRegistry`,
            already reflecting the caller's mutation.  Used here only for its
            ``len()`` (the manifest's ``key_count``) — the bytes on disk come
            from *payload*, not from re-serializing this object.
        payload: The exact bytes to write — the caller's own
            ``registry.save_bytes()`` call, serialized once and handed here.
        pre_sha: The tier's registry hash from BEFORE the caller's mutation —
            the caller's own pre-write read
            (:func:`~paramem.adapters.manifest.tier_registry_sha256`), always
            required.

    Returns:
        A :class:`RestampResult` carrying :func:`plan_restamp`'s outcome
        and (only for :data:`RESTAMPED`) the rebound slot path — or, when
        the rebind pair itself raised, :data:`REBIND_FAILED` with the
        exception's message in ``message``. Every status this function
        returns (never raises for) means the registry write already
        landed; only an uncaught registry-write ``OSError`` means it did
        not (see above).

    Raises:
        OSError: The registry write itself (``registry.save_from_bytes``)
            failed — nothing has landed. Never raised for a rebind-phase
            failure; that is returned as :data:`REBIND_FAILED` instead.
    """
    import hashlib as _hashlib
    from dataclasses import replace as _replace

    from paramem.adapters.manifest import ManifestError, read_manifest, write_manifest

    plan = plan_restamp(tier_root, registry=registry, payload=payload, pre_sha=pre_sha)

    registry.save_from_bytes(payload, tier_root / "indexed_key_registry.json")

    if plan.status != RESTAMPED:
        if plan.status in (KEYS_WITHOUT_SLOT, NO_PRE_WRITE_HASH, SLOT_ORPHANED):
            logger.warning(
                "restamp_tier_manifest: tier %s registry written but manifest "
                "not re-stamped (%s) -- recover via consolidation fold or "
                "registry restore",
                tier_root,
                plan.status,
            )
        return RestampResult(status=plan.status, slot=None)

    new_hash = _hashlib.sha256(payload).hexdigest()
    try:
        write_manifest(
            plan.slot,
            _replace(read_manifest(plan.slot), registry_sha256=new_hash, key_count=len(registry)),
        )
    except (OSError, ManifestError) as exc:
        # The registry write above already landed -- it precedes this
        # block unconditionally. Only the re-bind itself (the re-read of
        # the slot's current manifest, or the write of its replacement)
        # failed. Classified HERE, the only place that knows this failure
        # is the rebind and not the registry write: a caller that catches
        # this exact pair from this function can therefore rely on
        # REBIND_FAILED meaning "the registry mutation landed" as a
        # structural guarantee, not a claim it has to re-derive.
        logger.warning(
            "restamp_tier_manifest: tier %s registry written but slot rebind "
            "failed (%s) -- recover via consolidation fold or registry restore",
            tier_root,
            exc,
        )
        return RestampResult(status=REBIND_FAILED, slot=None, message=str(exc))
    return RestampResult(status=RESTAMPED, slot=plan.slot)


def erase_keys_and_restamp_manifest(
    *,
    adapter_dir: Path,
    keys: list[str],
) -> dict[str, RestampResult]:
    """Stale-mark *keys* directly against every tier's ON-DISK registry, keeping
    the registry file and (for a bound slot) the manifest in lockstep. This
    is the ruled erase door: there is no refusal concept, no precondition
    pre-pass and no auto-heal. A tier is *affected*, and a key is marked,
    iff that tier's registry holds the key ACTIVE — every affected tier
    reported in the return value has its mutation already landed, whatever
    the reported outcome, including a rebind failure — classified and
    reported by :func:`restamp_tier_manifest` itself as a distinct outcome,
    never raised (see Returns:). A tier that never reaches a reported
    outcome because ITS OWN registry write failed is not in the return
    value at all — that failure still propagates uncaught (see Raises:).

    A FILE SURGEON: every tier this function touches is discovered by walking
    *adapter_dir* (:func:`~paramem.memory.interim_adapter.iter_tier_roots`) and
    read via :meth:`~paramem.training.key_registry.KeyRegistry.load` — no
    :class:`~paramem.memory.store.MemoryStore` is constructed, hydrated, or
    read, and no model is touched. This is what lets the caller run the repair
    with no resident model (cloud-only mode) and while the live store is
    quarantined — there is nothing in RAM this function depends on.

    Per affected tier, in order:

    1. Read the pre-mutation registry hash
       (:func:`~paramem.adapters.manifest.tier_registry_sha256`) and load the
       tier's registry (:meth:`~paramem.training.key_registry.KeyRegistry.load`)
       — both for EVERY tier under *adapter_dir*, before any tier is mutated,
       so a tier whose on-disk registry exists but is not KeyRegistry-shaped
       (corrupt file, foreign schema) or cannot be decrypted (rotated key,
       corrupt age header) aborts before any mutation, for every affected
       tier at once — not mid-loop after an earlier tier has already
       committed.
    2. :meth:`~paramem.training.key_registry.KeyRegistry.stale` on each
       already-loaded registry that holds the key ACTIVE — withholds it,
       minting a marker that reserves the id and carries no fingerprint (the
       active fingerprint does not survive the transition). A key already
       withheld in a tier, or unknown to it, leaves that tier unaffected: no
       write and no rebind attempt for it. Entries and bookkeeping live in
       the (RAM-only) store, not on disk, and are untouched by this
       function — the row leaves with the rest of the key at the tier's own
       rebuild, not here; this door writes no ``key_metadata.json``. A
       tier's rebuild is a full consolidation or ``POST /reconsolidate``
       (both rebuild every main tier) or an interim cycle (rebuilds only the
       slot it mints) — a tier no consolidation reaches keeps its markers
       indefinitely.
    3. :func:`restamp_tier_manifest`, passing the step-1 pre-write hash and
       this call's own ``registry.save_bytes()`` payload — the no-retrain
       commit primitive: writes the mutated registry to disk, then plans
       and (when the plan names a slot) re-stamps the tier's bound slot so
       :func:`~paramem.adapters.manifest.find_live_slot` rebinds it on
       restart. This is an operator erase door: it NEVER refuses. This
       function does not classify that call's failures itself — it simply
       records whatever :class:`RestampResult` comes back. An ``OSError``
       or :class:`~paramem.adapters.manifest.ManifestError` raised while
       re-stamping THIS tier's slot manifest is caught INSIDE
       :func:`restamp_tier_manifest`, after its own registry write has
       already landed, and returned as :data:`REBIND_FAILED` (the
       exception's message carried in ``RestampResult.message``) — never
       raised from there, so this loop proceeds to the next affected tier
       without needing to catch anything itself. The same two exception
       shapes raised by the REGISTRY write inside that call — which
       precedes the classified block — are a different failure entirely
       (nothing for this tier has landed) and still propagate uncaught
       from here, aborting tiers ordered after the one that raised without
       unwinding tiers already committed ahead of it.

    A no-op — reads nothing, writes nothing, returns ``{}`` — when *keys* is
    empty. When *keys* is non-empty but every named key is already withheld
    (or unknown) in every tier, every tier's registry is still read (step 1)
    but none is affected — no bytes are written, no rebind is attempted, and
    this also returns ``{}``. This is designed idempotence, not a refusal:
    re-erasing an already-withheld key is an ordinary, silent success from
    this function's side; the caller reports what it named against what was
    already known.

    Callers that also hold a live, RAM-resident :class:`MemoryStore` for the
    same tiers (a healthy, non-quarantined server) are responsible for their
    own RAM-side sync (``store.discard_keys(keys)``) after this call returns
    — this function is disk-only and never reaches into a caller's store.

    Args:
        adapter_dir: Adapter store root the tier roots are resolved under
            (:func:`~paramem.memory.interim_adapter.iter_tier_roots`).
            Production caller passes ``config.adapter_dir``.
        keys: Indexed-memory keys to stale-mark. Production caller passes
            either the keys resolved for the target speaker via
            ``store.iter_bookkeeping()`` (``POST /speaker/forget``) or the
            operator's explicit list, filtered to registry-known keys
            (``POST /debug/erase-keys``).

    Returns:
        ``{tier_name: RestampResult}`` for every affected tier whose own
        registry write succeeded, so the caller can report per-tier what
        landed. Every tier named in this mapping — whatever its
        ``status`` — has its registry mutation landed on disk as a
        structural guarantee: :func:`restamp_tier_manifest` writes the
        registry unconditionally before it ever classifies a rebind
        outcome, so a status only reaches this mapping once that write is
        already done. :data:`RESTAMPED` and :data:`NOTHING_TO_BIND` are
        both a clean, bound outcome; :data:`KEYS_WITHOUT_SLOT`,
        :data:`NO_PRE_WRITE_HASH`, :data:`SLOT_ORPHANED`, and
        :data:`REBIND_FAILED` all mean the tier is left UNBOUND — logged
        at ERROR here, and it is the caller's job to surface it loudly
        (report it, never swallow it). Nothing here repairs an unbound
        tier automatically; the operator investigates and
        ``POST /backup/restore`` is the last resort. The first three of
        these four are unreachable on a healthy store: an "affected" tier's
        registry was just loaded, so its pre-write hash exists, and the
        verified steady state is digest-equals-bound-slot, so an orphaned
        or slotless shape only occurs on a store that is already broken
        and already loudly reported elsewhere. :data:`REBIND_FAILED` is the
        exception — reachable on an otherwise-healthy store from a
        transient I/O failure during the rebind write/read itself (disk
        full, permission change mid-request), not a sign the store was
        already broken.

    Raises:
        ValueError: A tier's on-disk registry file exists but is not
            KeyRegistry-shaped (propagated from
            :meth:`~paramem.training.key_registry.KeyRegistry.load`), raised
            before any tier is mutated.
        RuntimeError: Any tier under *adapter_dir* — not only one carrying
            a requested key — has an on-disk registry that exists but
            cannot be decrypted (propagated from
            :func:`~paramem.adapters.manifest.tier_registry_sha256`, called
            for every tier before affectedness is even determined; see step
            1 above). Raised before any tier is mutated — see the doc
            comment in :func:`~paramem.adapters.manifest.tier_registry_sha256`
            on why this caller deliberately does not catch it. There is
            nothing to stale-mark in a registry that cannot be read.

        An ``OSError`` raised by step 3's :func:`restamp_tier_manifest`
        call WHILE WRITING THAT TIER'S REGISTRY — as opposed to while
        re-stamping its slot manifest, which is classified inside that
        call and returned as :data:`REBIND_FAILED` instead (see step 3
        above and Returns: above) — propagates uncaught from here: that
        tier's mutation never landed, so there is nothing to name for it
        in the return value, and reporting it as any kind of "landed"
        outcome would be false. Aborts tiers ordered after the one that
        raised without unwinding tiers already committed ahead of it. Any
        other, non-I/O exception from that call propagates the same way.
    """
    if not keys:
        return {}

    from paramem.adapters.manifest import tier_registry_sha256
    from paramem.memory.interim_adapter import iter_tier_roots
    from paramem.training.key_registry import KeyRegistry

    affected: dict[str, tuple[Path, "KeyRegistry", str]] = {}
    for tier_name, tier_root in iter_tier_roots(adapter_dir):
        pre_sha = tier_registry_sha256(tier_root)
        registry = KeyRegistry.load(tier_root / "indexed_key_registry.json")
        if any(k in registry for k in keys):
            affected[tier_name] = (tier_root, registry, pre_sha)

    for _tier_root, registry, _pre_sha in affected.values():
        for key in keys:
            if key in registry:
                registry.stale(key)

    results: dict[str, RestampResult] = {}
    for tier_name, (tier_root, registry, pre_sha) in affected.items():
        # restamp_tier_manifest itself classifies a rebind-only failure as
        # REBIND_FAILED (see its own docstring): this loop just records
        # whatever status comes back. An OSError from the registry write
        # inside that call is a different failure -- it precedes any
        # classification and still propagates uncaught, aborting the
        # tiers ordered after this one; the registry mutation for a tier
        # that never reaches this line never landed.
        result = restamp_tier_manifest(
            tier_root,
            registry=registry,
            payload=registry.save_bytes(),
            pre_sha=pre_sha,
        )
        results[tier_name] = result
        if result.status in (RESTAMPED, NOTHING_TO_BIND):
            logger.info(
                "erase_keys_and_restamp_manifest: stale-marked key(s) in tier %s (file surgery)",
                tier_name,
            )
        else:
            logger.error(
                "erase_keys_and_restamp_manifest: tier %s left UNBOUND after registry "
                "mutation (%s) -- recover via a consolidation fold or registry restore",
                tier_name,
                result.status,
            )

    return results


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


def _write_tier_key_metadata(store, adapter_name: str, path: Path, tier_cycle: int) -> None:
    """Write *adapter_name*'s own bookkeeping rows to *path* (its ``key_metadata.json``).

    Called only from :func:`commit_tier_slot`.  One of the live-tier writers
    of ``key_metadata.json``: the fold/publish path writes the same file
    verbatim via :func:`publish_tier_registry`'s ``increment.rows_bytes``,
    and the restore path writes it from bundle bytes; this writer
    re-projects rows from the live store and serves only
    ``commit_tier_slot``'s own callers (migration, trial tree).  Scopes
    strictly to *adapter_name*'s own registry:
    ``store.registry(adapter_name).list_known()``
    (active ∪ stale — a staled key's bookkeeping must survive the
    active→stale transition), each row read from
    ``store.bookkeeping_for_key(key)`` verbatim.  Every known key already
    carries a row (the every-known-key-has-a-row invariant) — a gap is a
    violation to raise, never a case to write around: this writer must
    never persist a registry whose rows it just dropped.

    Writes ``{"tier_cycle": tier_cycle, "keys": {...}}`` — ``tier_cycle`` is
    the cycle at which this tier was last written (the loop's cycle counter at
    the moment of THIS commit).  ``cycle_count`` at boot is a derivation: the
    loop's counter is the maximum ``tier_cycle`` across every tier's file
    (see :func:`~paramem.server.consolidation.load_max_tier_cycle`) — there is
    no mutable global counter to flush.

    Args:
        store: The live :class:`~paramem.memory.store.MemoryStore`.
        adapter_name: The tier (or interim slot) whose rows are being
            written — the same name the registry is keyed under.
        path: Destination ``<tier_root>/key_metadata.json``.
        tier_cycle: The cycle to stamp as this tier's last-written cycle.

    Raises:
        ~paramem.memory.store.BookkeepingInvariantViolation: One or more of
            *adapter_name*'s known keys have no bookkeeping row in *store*.
    """
    from paramem.backup.encryption import write_infra_json
    from paramem.memory.store import raise_bookkeeping_invariant_violation

    keys_payload: dict = {}
    missing: list[str] = []
    for key in store.registry(adapter_name).list_known():
        bk = store.bookkeeping_for_key(key)
        if bk is None:
            missing.append(key)
            continue
        keys_payload[key] = dict(bk)
    if missing:
        raise_bookkeeping_invariant_violation(adapter_name, missing, "tier key metadata write")
    metadata = {"tier_cycle": tier_cycle, "keys": keys_payload}
    write_infra_json(path, metadata)


def commit_tier_slot(
    *,
    loop: "ConsolidationLoop",
    tier: str,
    adapter_name: str,
    stamp: str,
    mode: "Literal['simulate', 'train']",
    all_keyed: "list[dict]",
    output_dir: Path,
) -> Path:
    """The per-tier commit primitive for the store's own base-model swap and
    the trial-migration path — ``ConsolidationLoop.commit_main_tiers`` (both
    the trial tree's copy-forward and the live migration path in
    ``paramem.server.active_store_migration``) is its only production caller.

    A mini-fold's interim tier and a full fold's main-tier rebuild no longer
    go through this call: the two-phase event driver
    (``ConsolidationLoop.stage_event`` / ``run_build_and_publish``) commits
    each tier's slot via :func:`write_tier_slot` + :func:`publish_tier_registry`
    instead, parameterised by a :class:`TierIncrement` rather than *tier* /
    *adapter_name* / *stamp* / *all_keyed*. Registry is written last as
    the commit signal: its presence on disk means all preceding files
    (weights, simhash) are complete — the same ordering rule both commit
    paths share.

    Executes the full atomic commit sequence for one tier:

    1. ``loop.store.registry(adapter_name).save_bytes()`` — serialise the
       tier registry to canonical UTF-8 JSON bytes (no disk I/O).
    2. ``hashlib.sha256(payload)`` — hash the bytes so the manifest can stamp
       them before the registry is written to disk (pre-stamp invariant:
       manifest records the hash before the registry file exists).
    3. Build manifest — calls :func:`paramem.adapters.manifest.build_manifest_for`
       (train mode) or :func:`paramem.adapters.manifest.graph_payload_manifest`
       (simulate mode) with the ``registry_sha256_override``/``registry_sha256``
       and ``window_stamp=stamp``.  Simulate mode has no live PEFT model to
       hash, so its manifest carries no base-model/tokenizer/LoRA fingerprint
       (``None`` by construction — see :class:`~paramem.adapters.manifest.AdapterManifest`).
    4. Determine tier root via
       :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`.
       Same root for both modes — an interim slot dir, a main tier root
       (``<adapter_dir>/<tier>/``), or the equivalent path under a trial
       adapter tree when *output_dir* is a trial root.
    5. Write venue payload into a fresh timestamped slot under the tier
       root, through the one shared promotion sequence
       (:func:`~paramem.adapters.slot.write_slot`) — both venues return the
       promoted slot directory:

       - **Train mode**: :func:`paramem.models.loader.save_adapter`
         calls ``write_slot`` internally, writing the PEFT adapter weights
         plus the manifest (from step 3, embedded as ``meta.json``) into the
         slot.  Immediately after, a debug weight shadow is written for
         inspection/diff — ``with loop._artifact_scope():
         on_main_adapters_saved(loop.model, [adapter_name])`` — the per-adapter
         equivalent of what a whole-fold caller used to batch; the artifact
         scope resolves to no root when snapshots are off, so this is a no-op
         with no flag check.
       - **Simulate mode**: builds a ``MultiDiGraph`` from *all_keyed* (or,
         when *all_keyed* is empty, projects the tier fresh from
         ``loop.store`` via :func:`build_tier_graph_from_store`) and writes it
         into the slot as ``graph.json`` via ``write_slot`` (calling
         :func:`save_memory_to_disk` as its ``write_payload`` closure —
         encrypted/plaintext depending on daily-key state).  No PEFT weights
         are written, so there is nothing to shadow, but the slot itself is
         pruned exactly like a train-mode slot (step 8).

    6. Write this tier's own bookkeeping rows to
       ``<slot_root>/key_metadata.json`` — one whole-file write of *this
       tier's* keys only (``loop.store.registry(adapter_name).list_known()``,
       active ∪ stale), each row read from
       ``loop.store.bookkeeping_for_key(key)``, plus the tier-level
       ``tier_cycle`` field (``loop.cycle_count`` at the moment of this
       commit — the cycle this tier is being written at).  Ordered before the
       registry flush (step 7), same as the registry itself: rows first,
       registry last, so a newly-minted key's bookkeeping row is on disk
       before the registry write that makes the key discoverable on the next
       boot.  The fold path (interim and full-cycle tiers) writes the same
       file via :func:`publish_tier_registry`'s own
       ``increment.rows_bytes`` flush instead — this call's writer only
       serves ``commit_tier_slot``'s own callers (migration, trial tree).

    7. Flush the exact registry bytes from step 1 to
       ``<slot_root>/indexed_key_registry.json`` as the commit signal (both
       modes) via :meth:`paramem.training.key_registry.KeyRegistry.save_from_bytes`.
       This is the last write — its presence on disk signals that all preceding
       files are complete.
    8. Prune old slots (both modes, AFTER the commit signal so a caller
       reading disk mid-prune always sees a consistent (slot, registry) pair)
       — :func:`prune_old_slots`\\ ``(slot_root, <the slot step 5 wrote>,
       loop._keep_prior_slots)``.  ``prune_old_slots`` has live-slot immunity
       and filters via ``is_slot_name``, so interim siblings and registry
       files are never touched by it.

    Crash semantics: a kill after step 5 but before step 7 leaves the slot
    present without the registry file.  The boot-time keyless-tier sweep
    (:func:`paramem.server.app._sweep_keyless_tier_artifacts`) reads this
    shape via :func:`~paramem.adapters.registry_binding.verify_tier_binding`
    as :data:`~paramem.adapters.registry_binding.REGISTRY_ABSENT_WITH_SLOTS`
    — candidate slot(s) present, no ``indexed_key_registry.json`` at all —
    and that verdict is flagged LOUD: preserved and logged as an ERROR
    naming ``POST /backup/restore`` as the recovery door, never silently
    skipped and never reaped (a torn commit is not an interrupted erase, and
    the operator erase doors no longer produce interrupted erases of this
    shape at all). The mount
    loop in :func:`paramem.server.app._mount_adapters_from_slots` separately
    finds no matching slot for the interim tier's (degraded) hash and
    records its own ``registry_unverified`` row. The partial slot survives
    across restarts until an operator restores the registry or a later fold
    reuses or clears the slot.

    A *caught* exception (as opposed to the kill above) before the registry
    flush is cleaned up narrowly, never by removing *output_dir*'s tier root:
    only the ONE timestamped slot dir step 5 actually wrote this call
    (bound only once ``write_slot`` returns, in either venue) is removed; a
    failure before that call returns has nothing bound and removes nothing.
    A root freshly created by this call's own ``mkdir`` may be left behind
    (possibly empty); only the written slot is ever removed.

    Args:
        loop: The live :class:`paramem.training.consolidation.ConsolidationLoop`
            whose ``model``, ``tokenizer``, ``store``, and ``fingerprint_cache``
            are used for the write.
        tier: Tier name for registry lookup (e.g. ``"episodic"`` or
            ``"procedural"``).  Must be a tier registered in ``loop.store``.
        adapter_name: PEFT adapter name (e.g. ``"episodic_interim_YYYYMMDDTHHMM"``,
            ``"episodic"``, or ``"procedural"``).  Determines the on-disk slot
            root via :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`.
        stamp: Sub-interval or full-cycle window stamp (``"YYYYMMDDTHHMM"``)
            used as the ``window_stamp`` in the manifest and forwarded to
            :func:`build_manifest_for`.  Unused in simulate mode.
        mode: ``"train"`` writes adapter weights into the slot; ``"simulate"``
            writes a projected ``graph.json`` instead.
        all_keyed: The full list of keyed-pair dicts trained into (or
            simulated for) this tier slot.  Used by the simulate path to build
            the ``MultiDiGraph`` from the ground-truth list rather than by
            scanning the full store (avoids including keys from other tiers).
            An empty list falls back to a fresh :func:`build_tier_graph_from_store`
            projection of *adapter_name* from ``loop.store``.
        output_dir: Adapter store root (``loop.output_dir`` for the live
            store, or a trial/isolated adapter root for the trial-migration
            path).  Tier root is derived from this via ``adapter_slot_root_for_name``.

    Returns:
        The timestamped slot directory written — adapter weights (train,
        via :func:`~paramem.models.loader.save_adapter`) or the projected
        graph (simulate) — the same ``_written_slot`` this function's own
        crash-cleanup and prune-ordering already track internally.  Never
        ``None`` in either venue.  Raises on I/O failure.
    """
    import hashlib as _hashlib

    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    # Store keys are written under the adapter NAME (the go-live adoption
    # installs each increment under ``inc.tier`` == adapter name, see
    # MemoryStore.adopt_increments), so registry / simhash / graph
    # projections must read under the same key.
    # `tier` is retained for log messages and the on-disk slot hierarchy via
    # adapter_slot_root_for_name (which dispatches by adapter_name internally).
    # registry() always returns a KeyRegistry (setdefault auto-creates on first
    # access; never returns None) — no None check is needed here.
    tier_reg = loop.store.registry(adapter_name)

    # Empty commit is a real production path: len(tier_reg) == 0 is permitted
    # and produces an empty but valid registry file.  No zero-key guard is added.

    # --- Step 1+2: Serialize and hash (no disk I/O) ---
    payload = tier_reg.save_bytes()
    registry_sha256 = _hashlib.sha256(payload).hexdigest()

    slot_root = adapter_slot_root_for_name(output_dir, adapter_name)
    slot_root.mkdir(parents=True, exist_ok=True)

    # Track whether the registry flush (step 7 — the commit signal) has
    # completed.  Used in the finally block to remove the orphan SLOT on any
    # exception that fires before the flush.  After the flush the slot is
    # "live"; exceptions thereafter leave it intact, which is the documented
    # crash-safe outcome.
    _registry_flushed = False
    # The ONE timestamped slot dir this call's step 5 actually written
    # (weights or graph.json alike — both venues write through
    # paramem.adapters.slot.write_slot), bound only once that call returns.
    # `slot_root` is the adapter ROOT (prior slots, the registry file, and —
    # for episodic — interim_* siblings all live under it too), so cleanup
    # must remove this bound child, never slot_root itself.  Stays None only
    # when the failure happened before step 5 returned (nothing written yet
    # to remove).
    _written_slot: Path | None = None
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
            from paramem.models.loader import save_adapter as _save_adapter

            _written_slot = _save_adapter(loop.model, slot_root, adapter_name, manifest=manifest)
            logger.debug(
                "commit_tier_slot: adapter weights saved for %s (tier=%s)",
                adapter_name,
                tier,
            )

            # Debug weight shadow — per-adapter equivalent of the whole-fold
            # batch a caller used to run itself.  Resolves to a no-op when
            # snapshots are off (no flag check needed here).
            from paramem.utils.artifacts import on_main_adapters_saved

            with loop._artifact_scope():
                on_main_adapters_saved(loop.model, [adapter_name])
        else:
            # Simulate mode: build graph from all_keyed and write it into a
            # fresh timestamped slot through the same envelope the train
            # venue uses (paramem.adapters.slot.write_slot). When all_keyed is
            # empty, re-project from the canonical store so we do not
            # overwrite prior content with an empty graph.
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

            from paramem.adapters.manifest import (
                graph_payload_manifest as _graph_payload_manifest,
            )
            from paramem.adapters.slot import payload_filename as _payload_filename
            from paramem.adapters.slot import write_slot as _write_slot

            graph_manifest = _graph_payload_manifest(
                name=adapter_name,
                key_count=len(tier_reg),
                registry_sha256=registry_sha256,
                window_stamp=stamp,
            )

            def _write_graph_payload(pending_slot: Path, _graph: "nx.MultiDiGraph" = graph) -> None:
                save_memory_to_disk(_graph, pending_slot / _payload_filename("simulate"))

            _written_slot = _write_slot(
                slot_root, manifest=graph_manifest, write_payload=_write_graph_payload
            )
            logger.debug(
                "commit_tier_slot: graph slot written for %s (tier=%s, %d edges) -> %s",
                adapter_name,
                tier,
                graph.number_of_edges(),
                _written_slot,
            )

        # --- Step 6: Bookkeeping — durable BEFORE the registry flush (step 7) ---
        # The registry flush below is the commit signal that makes this
        # slot's keys discoverable on the next boot, so their bookkeeping
        # rows in key_metadata.json must already be on disk by the time that
        # signal lands.  A raise here fires before _registry_flushed is set,
        # so the finally block below removes the (still-orphan) written slot.
        # THIS TIER'S rows only — the per-tier commit primitive writes no
        # other tier's file.
        _write_tier_key_metadata(
            loop.store, adapter_name, slot_root / "key_metadata.json", loop.cycle_count
        )

        # --- Step 7: Registry flush — commit signal (both modes, LAST write) ---
        # The registry carries the tier's one fingerprint map (active keys
        # only -- a withheld id carries no fingerprint) in its "simhash" key,
        # so a separate simhash_registry.json is no longer written.
        # After this returns the slot is live on disk.  Any exception before
        # this point is caught by the finally block below which removes the
        # orphan written slot (both modes — see the block's docstring).
        tier_reg.save_from_bytes(
            payload,
            slot_root / "indexed_key_registry.json",
        )
        _registry_flushed = True

        # --- Step 8: Prune old slots — AFTER the commit signal, both modes
        # (simulate slots accumulate under the tier root exactly like train
        # slots do, and are pruned the same way).
        # By this point _written_slot is always bound: both venue arms of
        # step 5 (above) assign it unconditionally on return — it is None
        # only when an exception fires before step 5 returns, and that
        # exception would have propagated out of this try block long before
        # reaching step 7's registry flush above, never landing here.
        assert _written_slot is not None
        prune_old_slots(slot_root, _written_slot, loop._keep_prior_slots)
    finally:
        if not _registry_flushed and _written_slot is not None:
            # The commit signal was never written.  Remove ONLY the slot dir
            # this call actually wrote — never slot_root itself, which also
            # holds prior slots, the registry file, and (for episodic) the
            # interim_* siblings.  Nothing to remove when the failure
            # happened before step 5 returned (nothing bound yet).
            # Best-effort: cleanup errors are swallowed (ignore_errors=True) so
            # the original exception propagates unmodified to the caller.
            import shutil as _shutil

            _shutil.rmtree(_written_slot, ignore_errors=True)
    logger.info(
        "commit_tier_slot: committed %s (tier=%s, mode=%s, keys=%d)",
        adapter_name,
        tier,
        mode,
        len(tier_reg),
    )
    return _written_slot


# ---------------------------------------------------------------------------
# Two-phase write / publish primitives — one act each, driven by a build
# already assembled off-store (paramem.memory.increment.TierIncrement).
# Neither primitive consults the live store: the increment already carries
# the registry, the rows and the keyed list.  Neither writes a ledger entry —
# the driver hashes the artifacts these primitives return/write and records
# the stage entry itself, so this module stays free of event state.
# ---------------------------------------------------------------------------


def write_tier_slot(
    *,
    ctx: "TierWriteContext",
    increment: "TierIncrement",
    stamp: str,
    mode: "Literal['simulate', 'train']",
) -> Path:
    """Write one increment's PAYLOAD.  Returns the slot directory written.

    Both venues write into a fresh timestamped slot under *ctx.output_dir*'s
    resolved tier root, through the one shared promotion sequence
    (:func:`~paramem.adapters.slot.write_slot`):

    train:    payload = ``atomic_save_adapter(ctx.model, slot_root,
              STAGING_ADAPTER, manifest=<build_manifest_for(...,
              STAGING_ADAPTER, ...) stamped with
              sha256(increment.registry_bytes), then ``name`` replaced by
              ``increment.adapter_name``>)`` — ``atomic_save_adapter`` calls
              ``write_slot`` internally.
    simulate: payload = ``graph.json`` projected from ``increment.keyed``,
              manifest = :func:`~paramem.adapters.manifest.graph_payload_manifest`
              stamped the same way — this function calls ``write_slot``
              directly.

    Call INSIDE ``staged_weights`` (train mode), after the gate, before the
    staging slot is disposed, and ONLY for a member that carries a payload:
    ``increment.has_payload`` (``increment.rebuilt`` AND ``increment.keyed``
    non-empty).  A tier rebuilt to zero keys has a keyed list and nothing to
    train, so it never reaches here; its ``tier_written`` entry hashes its
    shadow artifact set instead (the caller's responsibility, not this
    function's).

    Writes NOTHING the live store reads, in EITHER venue: the written slot is
    inert until its ``registry_sha256`` stamp matches the registry bytes on
    disk — :func:`~paramem.adapters.manifest.find_live_slot` only then binds
    it — which :func:`publish_tier_registry` is what lands. NO registry
    write, NO rows write, NO PRUNING here — prior slots are pruned by the
    go-live sequencer after the publish, because before the publish the
    bound slot is still the OLD one.

    The debug weight shadow (:func:`~paramem.utils.artifacts.on_main_adapters_saved`)
    is called unconditionally, mirroring :func:`commit_tier_slot`'s own
    inline debug-snapshot call — no flag check here or there, per the
    debug-hook contract (:func:`~paramem.utils.artifacts.debug_run`:
    "callers never test the flag themselves"). The write it performs is
    gated entirely by whether an artifact scope is currently active
    (``loop._artifact_scope()``, opened by whichever caller up the stack
    chose to); when none is, the call is a documented no-op. A resumed
    dispatch never opens one (:func:`~paramem.server.app._run_pending_event_resume`
    calls straight into the two-phase driver with no scope), so this
    call is always inert on that path — same outcome as debug being off,
    reached through the scope gate rather than a local flag. The
    base-model fingerprint comes from ``ctx.fingerprint_cache``.  There is no
    ``verify=`` callback: the post-save recall probe is retired — the
    recall gate that authorises a write runs once, on the staged weights,
    immediately before this call.

    Args:
        ctx: The write context.  Reads ``model``, ``tokenizer``,
            ``fingerprint_cache``, ``output_dir``.
        increment: The increment to write.  Must satisfy ``has_payload``.
        stamp: ``ledger["stamp"]`` — the window/interim stamp, recorded as
            the manifest's ``window_stamp``.
        mode: ``"train"`` writes adapter weights; ``"simulate"`` writes the
            projected ``graph.json``.

    Returns:
        The timestamped slot directory written — adapter weights (train) or
        the projected graph (simulate). Never a bare file path in either
        venue.

    Raises:
        ValueError: *increment* carries no payload to write
            (``not increment.has_payload``).
    """
    import hashlib as _hashlib

    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    if not increment.has_payload:
        raise ValueError(
            f"write_tier_slot: increment for tier {increment.tier!r} carries no "
            "payload to write (rebuilt=False or an empty keyed list)"
        )

    slot_root = adapter_slot_root_for_name(ctx.output_dir, increment.adapter_name)
    slot_root.mkdir(parents=True, exist_ok=True)

    if mode == "train":
        from dataclasses import replace as _replace

        from paramem.adapters.manifest import build_manifest_for as _build_manifest_for
        from paramem.training.trainer import STAGING_ADAPTER

        registry_sha256 = _hashlib.sha256(increment.registry_bytes).hexdigest()
        manifest = _build_manifest_for(
            ctx.model,
            ctx.tokenizer,
            STAGING_ADAPTER,
            key_count=len(increment.registry),
            base_model_hash_cache=ctx.fingerprint_cache,
            registry_sha256_override=registry_sha256,
            window_stamp=stamp,
            adapter_root=ctx.output_dir,
        )
        # build_manifest_for reads shape from model.peft_config[STAGING_ADAPTER]
        # (the only adapter guaranteed resident at write time) but the
        # recorded `name` must be the TIER's adapter name — the promote at
        # go-live is what makes the tier adapter carry this event's weights.
        manifest = _replace(manifest, name=increment.adapter_name)

        from paramem.models.loader import save_adapter as _save_adapter

        written = _save_adapter(ctx.model, slot_root, STAGING_ADAPTER, manifest=manifest)
        logger.info(
            "write_tier_slot: written tier %s from staging adapter -> %s",
            increment.tier,
            written,
        )

        from paramem.utils.artifacts import on_main_adapters_saved

        on_main_adapters_saved(ctx.model, [increment.adapter_name])

        return written

    # simulate mode: project a graph from the increment's keyed list and
    # write it through the same slot envelope the train venue uses.
    from paramem.adapters.manifest import graph_payload_manifest as _graph_payload_manifest
    from paramem.adapters.slot import payload_filename as _payload_filename
    from paramem.adapters.slot import write_slot as _write_slot

    graph = nx.MultiDiGraph()
    for kp in increment.keyed:
        _add_keyed_edge(
            graph,
            kp["subject"],
            kp["object"],
            indexed_key=kp["key"],
            predicate=kp.get("predicate", ""),
            speaker_id=kp.get("speaker_id", ""),
        )
    registry_sha256 = _hashlib.sha256(increment.registry_bytes).hexdigest()
    manifest = _graph_payload_manifest(
        name=increment.adapter_name,
        key_count=len(increment.registry),
        registry_sha256=registry_sha256,
        window_stamp=stamp,
    )

    def _write_graph_payload(pending_slot: Path, _graph: "nx.MultiDiGraph" = graph) -> None:
        save_memory_to_disk(_graph, pending_slot / _payload_filename("simulate"))

    written = _write_slot(slot_root, manifest=manifest, write_payload=_write_graph_payload)
    logger.info(
        "write_tier_slot: written simulate graph for tier %s -> %s (%d edges)",
        increment.tier,
        written,
        graph.number_of_edges(),
    )
    return written


_WRITTEN_SLOT_MISSING: Final[str] = "slot_missing"
_WRITTEN_MANIFEST_UNREADABLE: Final[str] = "manifest_unreadable"
_WRITTEN_SLOT_INCOMPLETE: Final[str] = "slot_incomplete"
_WRITTEN_REGISTRY_HASH_MISMATCH: Final[str] = "registry_hash_mismatch"


def _written_member_refusal(slot: Path, increment: "TierIncrement") -> "str | None":
    """Pure-read publish-precondition check for one WRITTEN bundle member.

    Called by :func:`assert_publish_preconditions` for every bundle member
    whose ``written_slots`` entry is not ``None``. Four checks, in order:

    1. The recorded slot exists and is a directory.
    2. Its manifest parses (:func:`~paramem.adapters.manifest.read_manifest`).
    3. Every name :func:`~paramem.adapters.slot.required_slot_files` names
       for its ``payload.kind`` is present in the slot.
    4. The manifest's ``registry_sha256`` already equals
       ``sha256(increment.registry_bytes)`` — the bytes about to land.
       Landing registry bytes whose digest the written manifest does not
       carry would orphan the slot the moment the registry lands.

    No payload re-hash: :func:`~paramem.adapters.slot.write_slot` computed
    ``payload.sha256`` over the bytes it had just written when it written
    this slot, and re-hashing the payload here would be a second
    invocation of a transformation the write already owns. Payload-content
    drift is the boot-time binding verification's question, not this
    preflight's.

    Args:
        slot: The recorded written-slot path for this member
            (``written_slots[increment.tier]``, already known not ``None``).
        increment: The bundle member being checked.

    Returns:
        ``None`` when the member is publishable; otherwise one of
        :data:`_WRITTEN_SLOT_MISSING`, :data:`_WRITTEN_MANIFEST_UNREADABLE`,
        :data:`_WRITTEN_SLOT_INCOMPLETE`, :data:`_WRITTEN_REGISTRY_HASH_MISMATCH`.
    """
    import hashlib as _hashlib

    from paramem.adapters.manifest import ManifestError, read_manifest
    from paramem.adapters.slot import required_slot_files

    if not slot.is_dir():
        return _WRITTEN_SLOT_MISSING
    try:
        manifest = read_manifest(slot)
    except ManifestError:
        return _WRITTEN_MANIFEST_UNREADABLE
    for name in required_slot_files(manifest.payload.kind):
        if not (slot / name).exists():
            return _WRITTEN_SLOT_INCOMPLETE
    if manifest.registry_sha256 != _hashlib.sha256(increment.registry_bytes).hexdigest():
        return _WRITTEN_REGISTRY_HASH_MISMATCH
    return None


def assert_publish_preconditions(
    *,
    bundle: "Sequence[TierIncrement]",
    ctx: "TierWriteContext",
    written_slots: "Mapping[str, Path | None]",
) -> None:
    """Validate every bundle member's publish preconditions before the first byte lands.

    The FIRST statement of :func:`~paramem.training.go_live.publish_bundle`'s
    publish act — before :func:`publish_tier_registry`'s own ``tier_root.mkdir``
    and its rows write, the first durable write of the sequence. Performs
    PURE READS only (``Path.exists``, ``Path.is_dir``,
    :func:`~paramem.adapters.manifest.read_manifest`,
    :func:`~paramem.adapters.manifest.count_slot_candidates`,
    :func:`~paramem.adapters.manifest.find_live_slot`, and in-memory
    ``sha256``) — no ``mkdir``, no open-for-write, no rename — so a refusal
    from this function leaves the whole on-disk tree byte-for-byte
    unchanged.

    Two precondition sets, one per member shape:

    * A WRITTEN member (``written_slots[increment.tier] is not None``) —
      :func:`_written_member_refusal`.
    * A ROWS-ONLY member (``written_slots[increment.tier] is None``) —
      :func:`plan_restamp`, the SAME planner :func:`restamp_tier_manifest`
      composes at write time; a plan status other than :data:`RESTAMPED` or
      :data:`NOTHING_TO_BIND` refuses.

    Raises :class:`TierWriteRefused` naming EVERY failing member — the loop
    over the whole bundle completes before raising, so a caller sees every
    problem at once rather than one refusal per retry. Not caught by
    :func:`~paramem.training.go_live.publish_bundle`; it propagates through
    :meth:`~paramem.training.consolidation.ConsolidationLoop.run_build_and_publish`
    into the dispatcher's crash envelope, loud.

    State after a refusal: the stage ledger stays pending with its
    ``tier_written`` entries intact, the shadow tree is untouched, no tier
    goes live, and every contributing session stays pending (retirement
    happens only at the caller's terminal on ``all_live``). The event
    resumes after the operator's repair and re-writes nothing that already
    written.

    Zero-bytes-on-refusal is the automated publish path's property only —
    the operator erase doors (:func:`erase_keys_and_restamp_manifest`) have
    no refusal concept at all: they always land the mutation and always
    attempt the rebind.

    Args:
        bundle: The ordered go-live bundle about to publish.
        ctx: The write context — ``output_dir`` is read to resolve each
            rows-only member's tier root.
        written_slots: ``tier -> write_tier_slot's return`` for every bundle
            member — the same mapping
            :func:`~paramem.training.go_live.publish_bundle` was itself
            called with.

    Raises:
        TierWriteRefused: One or more bundle members fail their
            precondition set. ``refusals`` names every failing tier.
    """
    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    refusals: dict[str, str] = {}
    for increment in bundle:
        written_slot = written_slots.get(increment.tier)
        if written_slot is not None:
            reason = _written_member_refusal(written_slot, increment)
            if reason is not None:
                refusals[increment.tier] = reason
            continue

        tier_root = adapter_slot_root_for_name(ctx.output_dir, increment.adapter_name)
        plan = plan_restamp(
            tier_root,
            registry=increment.registry,
            payload=increment.registry_bytes,
            pre_sha=increment.pre_sha,
        )
        if plan.status not in (RESTAMPED, NOTHING_TO_BIND):
            refusals[increment.tier] = plan.status

    if refusals:
        raise TierWriteRefused(refusals)


def publish_tier_registry(
    *,
    increment: "TierIncrement",
    ctx: "TierWriteContext",
    written_slot: "Path | None",
) -> None:
    """Write the tier's rows, then its registry payload.  THE commit signal.

    ``key_metadata.json`` first, ``indexed_key_registry.json`` last — one
    ordering rule shared with the restore path. BOTH files are written from
    the increment's verbatim payloads (``rows_bytes``, ``registry_bytes``),
    never from a re-serialization of the parsed objects: the manifest binds
    by a digest of those exact bytes, so byte-stability across a process
    boundary is a correctness condition, not a nicety.

    Always called AFTER :func:`assert_publish_preconditions` has already
    validated the whole bundle (:func:`~paramem.training.go_live.publish_bundle`'s
    ordering), so this function's own writes are expected to always
    succeed cleanly.

    ``written_slot is not None``: flush ``increment.registry_bytes`` through
    :meth:`~paramem.training.key_registry.KeyRegistry.save_from_bytes`. Those
    are the bytes hashed into that slot's manifest, so
    :func:`~paramem.adapters.manifest.find_live_slot` binds it the moment
    they land.

    ``written_slot is None``: route the registry write through
    :func:`restamp_tier_manifest` (``registry=increment.registry,
    payload=increment.registry_bytes, pre_sha=increment.pre_sha``), which
    saves the payload AND, when its plan names a slot, re-stamps that
    slot's manifest to its digest in one call.  Two calls would leave the
    tier bound to nothing in between. A returned status other than
    :data:`~paramem.memory.persistence.RESTAMPED` or
    :data:`~paramem.memory.persistence.NOTHING_TO_BIND` is raised here as
    :class:`TierWriteRefused` — this covers two different situations under
    one channel: a plan-level refusal (a structural guard only; on the
    publish path this is unreachable, since the preflight already checked
    every member, so seeing one here means a concurrent mutation changed
    the tier underneath this event) and :data:`REBIND_FAILED` (an actual
    I/O failure during the rebind itself, reachable even with a clean
    preflight — the registry write has already landed either way, loud
    after a landed write being the correct severity for both).
    :func:`restamp_tier_manifest` itself never raises for a rebind
    failure — only a registry-write ``OSError`` propagates uncaught from
    it, and that propagates uncaught from here too rather than becoming a
    :class:`TierWriteRefused`, since nothing landed for this tier at all.

    Prunes nothing — the go-live sequencer prunes prior slots after the
    mount.

    Args:
        increment: The increment being published.
        ctx: The write context.  Reads ``output_dir`` only.
        written_slot: :func:`write_tier_slot`'s return, or ``None`` for a
            member that written no payload (a rows-only member, or a
            zero-key rebuild).

    Raises:
        TierWriteRefused: The no-retrain restamp path (``written_slot is
            None``) returned a status other than ``RESTAMPED``/
            ``NOTHING_TO_BIND`` for this tier — either a plan-level
            refusal (a structural guard for a caller that reached this
            function without running :func:`assert_publish_preconditions`
            first; unreachable on the publish path itself) or
            :data:`REBIND_FAILED` (a transient I/O failure during the
            rebind; reachable even on the publish path, since the
            preflight cannot predict a failure that has not happened yet).
        OSError: The registry write itself, inside
            :func:`restamp_tier_manifest`, failed — nothing landed for
            this tier. Propagates uncaught rather than becoming a
            :class:`TierWriteRefused`.
    """
    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    tier_root = adapter_slot_root_for_name(ctx.output_dir, increment.adapter_name)
    tier_root.mkdir(parents=True, exist_ok=True)

    rows_path = tier_root / "key_metadata.json"
    from paramem.backup.encryption import write_infra_bytes

    write_infra_bytes(rows_path, increment.rows_bytes)

    if written_slot is not None:
        registry_path = tier_root / "indexed_key_registry.json"
        increment.registry.save_from_bytes(increment.registry_bytes, registry_path)
        logger.info(
            "publish_tier_registry: published tier %s (written slot %s)",
            increment.tier,
            written_slot,
        )
        return

    result = restamp_tier_manifest(
        tier_root,
        registry=increment.registry,
        payload=increment.registry_bytes,
        pre_sha=increment.pre_sha,
    )
    if result.status not in (RESTAMPED, NOTHING_TO_BIND):
        raise TierWriteRefused({increment.tier: result.status})
    logger.info(
        "publish_tier_registry: published tier %s via restamp (status=%s)",
        increment.tier,
        result.status,
    )


def prune_old_slots(tier_root: Path, live_slot: Path, keep: int) -> None:
    """Remove prior slots beyond the retention budget.

    Reads no loop state.  Called by :func:`commit_tier_slot` itself
    (step 8, after the registry flush, both venues — a simulate slot
    accumulates under the tier root exactly like a train slot and is pruned
    the same way), by the go-live sequencer after a member's publish, and by
    the donor build (``paramem.training.donor.build_donor``, at ``keep=0``).

    Args:
        tier_root: Slot-kind directory (e.g. ``<adapter_dir>/episodic/``).
        keep: Max number of non-live prior slots to retain (``>= 0``).
        live_slot: The slot just published for this tier; immune to
            pruning whatever *keep* says.

    IMMUNE IN ADDITION to *live_slot*, whatever *keep* says: whatever
    ``find_live_slot(tier_root, tier_registry_sha256(tier_root))`` binds
    right now — but ONLY when that digest is non-empty. ``keep=0`` is a
    legal setting, so without this immunity the retention pass could delete
    the slot that is currently serving. An empty digest means the root binds
    nothing: :func:`~paramem.adapters.manifest.tier_registry_sha256` returns
    ``""`` for a root with no registry and
    :func:`~paramem.adapters.manifest.find_live_slot` matches ``""`` against
    every slot stamped ``""`` — the donor store's exact shape (no registry,
    manifests stamped ``registry_sha256_override=""``) — so the donor
    build's ``keep=0`` still leaves exactly one donor artifact.
    """
    import shutil as _shutil

    from paramem.adapters.manifest import find_live_slot, is_slot_name, tier_registry_sha256

    tier_root = Path(tier_root)
    if not tier_root.is_dir() or keep < 0:
        return

    immune = {live_slot}
    live_digest = tier_registry_sha256(tier_root)
    if live_digest:
        bound = find_live_slot(tier_root, live_digest)
        if bound is not None:
            immune.add(bound)

    candidates: list[Path] = []
    for entry in tier_root.iterdir():
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            continue
        if entry in immune:
            continue
        if not is_slot_name(entry.name):
            continue
        candidates.append(entry)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in candidates[keep:]:
        _shutil.rmtree(stale, ignore_errors=False)
        logger.info("prune_old_slots: removed %s (retention=%d)", stale, keep)
