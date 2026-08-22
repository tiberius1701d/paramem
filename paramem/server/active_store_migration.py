"""Migrate keyed facts between train (LoRA weights) and simulate (graph.json) stores.

Triggered when the operator flips ``consolidation.mode`` in server.yaml.
The migration is per-store with a 1.0 recall gate before cleanup; the
source store stays authoritative until ALL stores have completed and the
state file is removed. On crash mid-migration, the state file persists
and the system retains the source mode on next boot.

State file location: ``<paths.adapters>/.active_store_migration.json``
(age-encrypted via ``write_infra_bytes`` when the daily identity is loaded).

Both venues write into the same shape: a fresh timestamped slot directory
under the tier root (``<adapter_dir>/<tier>/<ts>/``), written through the
shared slot envelope (:func:`~paramem.adapters.slot.write_slot`) — a train
slot carries ``adapter_model.safetensors``, a simulate slot carries
``graph.json``. The bound slot for a tier is resolved the same way in
either venue: :func:`~paramem.adapters.manifest.find_live_slot` against
:func:`~paramem.adapters.manifest.tier_registry_sha256`. The distinction
between modes is which payload kind the bound slot's manifest declares
(``manifest.payload.kind``), never a flat filename check at the tier root.

Two directions:

* ``simulate_to_train``: read the tier's bound simulate slot's ``graph.json``
  → train into a staging slot → recall probe the staged weights at
  ``loop.config.recall_sanity_threshold`` → on pass, promote into
  ``<name>`` and durably commit the tier slot (weights, bookkeeping, and
  registry, in that order) via :func:`~paramem.memory.persistence.commit_tier_slot`.
  On fail, ``<name>`` is never touched (it stays at LoRA-zero) and the source
  simulate slot is left intact.

* ``train_to_simulate``: resolve the tier's bound train slot; when no bound
  simulate slot already covers the current active keys, reconstruct the
  graph from weights and write it into a fresh simulate slot through the same
  envelope the train venue uses → delete the train weight slot(s). On fail,
  nothing is written (the sanity check runs before the write, not after — see
  :func:`_migrate_tier_train_to_simulate`) and the train slot is left intact.

Per-store failures are recorded in the state file but do not abort the
remaining stores — the operator can re-trigger to retry.

Migration relocates every registered store (main tiers: episodic, semantic,
procedural; plus any loaded interim stores such as
``episodic_interim_<stamp>``) under its own identity.  Interim adapters use
the episodic LoRA config and their on-disk slot root is resolved via
``adapter_slot_root_for_name`` so the hierarchy under
``<adapter_dir>/episodic/interim_<stamp>/`` is honoured.

``detect_mode_switch`` arms on main-tier shape only (the three main-tier
directories are the canonical signal for a mode mismatch); ``migrate()``
then relocates the full set of registered stores including any interim slots
that were loaded at boot.  Do not change ``detect_mode_switch`` to inspect
interim dirs — it would produce false positives on partially-consolidated
systems.

``detect_mode_switch`` classifies each main tier's payload by reading its
bound slot's manifest (``payload.kind``, via ``find_live_slot`` +
``read_manifest``) — never a sibling interim slot's own bound slot, since
each main tier's bound slot is resolved against that tier's own root.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes
from paramem.training.consolidation import RecallGateRejected
from paramem.training.trainer import STAGING_ADAPTER, promote_staging_adapter, staged_weights
from paramem.utils.tiers import MAIN_TIERS

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig
    from paramem.training.consolidation import ConsolidationLoop

logger = logging.getLogger(__name__)


_STATE_FILENAME = ".active_store_migration.json"


# ---------------------------------------------------------------------------
# State file model
# ---------------------------------------------------------------------------


@dataclass
class MigrationState:
    """Persisted state of an in-progress active-store migration.

    The file's presence on disk is the signal that a migration was started
    and not completed. Startup detection treats this as "the source store
    stays authoritative until all stores complete".
    """

    direction: str  # "simulate_to_train" | "train_to_simulate"
    started_at: str  # iso8601
    source_mode: str  # "simulate" | "train" — source store until all stores complete
    target_mode: str  # "simulate" | "train" — what the operator's yaml asks for
    completed_tiers: list[str] = field(default_factory=list)
    failed_tiers: dict[str, str] = field(default_factory=dict)  # tier -> error msg

    @classmethod
    def for_mode_switch(cls, *, source_mode: str, target_mode: str) -> "MigrationState":
        if source_mode == target_mode:
            raise ValueError(f"source_mode and target_mode are both {source_mode!r}")
        if source_mode not in ("simulate", "train") or target_mode not in ("simulate", "train"):
            raise ValueError(
                f"modes must be 'simulate' or 'train', got "
                f"source={source_mode!r} target={target_mode!r}"
            )
        return cls(
            direction=f"{source_mode}_to_{target_mode}",
            started_at=datetime.now(timezone.utc).isoformat(),
            source_mode=source_mode,
            target_mode=target_mode,
        )

    def all_tiers_done(self, registered_tiers: list[str]) -> bool:
        """Return True when every registered store has been relocated cleanly.

        Args:
            registered_tiers: Live set of store names to check, obtained from
                ``loop.store.tiers_with_registry()``.  Covers main tiers
                (episodic, semantic, procedural) plus any loaded interim stores
                (e.g. ``episodic_interim_<stamp>``).  The set is NOT persisted
                — it is passed by the caller at check-time so that an in-flight
                state file from a prior run never silently ignores newly-registered
                interim stores that were added between boot and the check.

        Returns:
            ``True`` only when all names in *registered_tiers* appear in
            ``completed_tiers`` and ``failed_tiers`` is empty.
        """
        return all(t in self.completed_tiers for t in registered_tiers) and not self.failed_tiers


def state_path(adapter_dir: Path) -> Path:
    return Path(adapter_dir) / _STATE_FILENAME


def load_state(adapter_dir: Path) -> Optional[MigrationState]:
    """Read the state file. Returns None when absent or unreadable."""
    p = state_path(adapter_dir)
    if not p.exists():
        return None
    try:
        raw = read_maybe_encrypted(p).decode("utf-8")
        return MigrationState(**json.loads(raw))
    except Exception:
        logger.exception("Failed to load active-store migration state at %s", p)
        return None


def save_state(adapter_dir: Path, state: MigrationState) -> None:
    """Write the state file (age-encrypted at rest when daily identity loaded)."""
    p = state_path(adapter_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_infra_bytes(p, json.dumps(asdict(state), indent=2).encode("utf-8"))


def clear_state(adapter_dir: Path) -> None:
    """Remove the state file — the signal that migration completed cleanly."""
    p = state_path(adapter_dir)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _has_adapter_registry(adapter_dir: Path, tier: str) -> bool:
    """Return True when a per-tier indexed_key_registry.json exists for *tier*.

    The canonical layout is ``<adapter_dir>/<tier>/indexed_key_registry.json``.
    This is the commit signal written last by ``commit_tier_slot``: its
    presence on disk means all preceding adapter files are complete.
    """
    return (Path(adapter_dir) / tier / "indexed_key_registry.json").exists()


def detect_mode_switch(config: "ServerConfig") -> Optional[MigrationState]:
    """Detect if the active-store state diverges from ``config.consolidation.mode``.

    Detection logic:

    1. If a state file exists, return it (migration was started, possibly
       interrupted, must be resumed before inference is consistent).
    2. Otherwise compare each main tier's actual on-disk payload — resolved
       via its own bound slot
       (:func:`~paramem.adapters.manifest.find_live_slot` against
       :func:`~paramem.adapters.manifest.tier_registry_sha256`, read back
       via :func:`~paramem.adapters.manifest.read_manifest`'s
       ``payload.kind``; a main tier with no bound slot contributes ``None``,
       and a bound slot is always resolved against that tier's own root, so
       a sibling interim slot's payload never satisfies it) — to the
       operator's yaml ``mode``:

       * ``mode=train`` and a main tier carries a ``"simulate"`` payload but
         none carries ``"train"`` → ``simulate_to_train`` migration is needed.
       * ``mode=simulate`` and a main tier carries a ``"train"`` payload but
         none carries ``"simulate"`` → ``train_to_simulate`` migration is
         needed.

    Returns ``None`` when the active store is consistent with the mode
    (no migration needed).
    """
    from paramem.adapters.manifest import (
        ManifestNotFoundError,
        ManifestSchemaError,
        find_live_slot,
        read_manifest,
        tier_registry_sha256,
    )

    existing = load_state(config.adapter_dir)
    if existing is not None:
        return existing

    target_mode = config.consolidation.mode
    if target_mode not in ("simulate", "train"):
        return None  # unsupported mode — let upstream complain

    adapter_dir = Path(config.adapter_dir)
    payload_kinds: list["str | None"] = []
    for t in MAIN_TIERS:
        tier_root = adapter_dir / t
        bound = find_live_slot(tier_root, tier_registry_sha256(tier_root))
        kind: "str | None" = None
        if bound is not None:
            try:
                kind = read_manifest(bound).payload.kind
            except (ManifestNotFoundError, ManifestSchemaError):
                # Race: meta.json removed/corrupted between find_live_slot's
                # scan and this read — treat as no bound payload, same as
                # slot=None, rather than raising out of a detection pass.
                kind = None
        payload_kinds.append(kind)
    simulate_present = "simulate" in payload_kinds
    adapter_present = "train" in payload_kinds

    if target_mode == "train" and simulate_present and not adapter_present:
        return MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
    if target_mode == "simulate" and adapter_present and not simulate_present:
        return MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")

    return None


# ---------------------------------------------------------------------------
# Migration execution
# ---------------------------------------------------------------------------


class _TierSkipped(Exception):
    """Tier has no source data; advance state without recording failure."""


def migrate(
    loop: "ConsolidationLoop", config: "ServerConfig", state: MigrationState
) -> MigrationState:
    """Execute the active-store migration described by *state*.

    Per-store; each store's success is persisted to the state file before
    moving to the next, so a crash mid-migration can resume from the
    last committed store on the next call.  The set of stores to migrate is
    read from ``loop.store.tiers_with_registry()`` at call time and covers
    main tiers (episodic, semantic, procedural) plus any interim stores
    loaded at boot (e.g. ``episodic_interim_<stamp>``).

    Source store is preserved until ALL registered stores have completed
    cleanly (``state.all_tiers_done(registered_tiers)``) — only then is the
    state file removed and the source-side artifacts deleted.

    This migration is content-PRESERVING (it re-encodes the same keys in
    the other venue, never replacing what a pending event derived
    against), so it requires a coherent tree: when a consolidation event's
    stage ledger is still pending, this function refuses outright — see
    the ``RuntimeError`` below — before touching the state file, so the
    pending record, its extraction tree, and all scratch stay exactly as
    they were and the effective mode is not flipped.  A pending event must
    resume to completion (publishing per its own ledger's recorded venue)
    before a migration can run against the tree it leaves behind.

    Raises:
        RuntimeError: When ``loop.store.tiers_with_registry()`` returns an
            empty list BUT on-disk source content exists.  This indicates that
            the boot-time registry load failed (silent swallow upstream) and
            the in-memory store does not reflect the on-disk state.  Proceeding
            would cause ``all_tiers_done([])`` to vacuously return ``True``,
            ``clear_state`` to fire, and ``_finalize_migration`` to flip the
            effective mode — a silent data-loss path.  The state file is NOT
            cleared so the migration stays pending and is surfaced on retry
            after the operator resolves the corrupt registry.
        RuntimeError: When a consolidation event's stage ledger is still
            pending.  Raised before ``save_state`` runs, so a first-time
            mode-switch trigger never even creates the migration state
            file — the caller resumes the pending event (``POST
            /consolidate`` resumes it; the schedule otherwise resumes it
            on its own next dispatch) and this function is called again
            once the tree is coherent.

    Returns the updated state.
    """
    from paramem.memory.interim_adapter import iter_interim_dirs

    registered_tiers = loop.store.tiers_with_registry()

    if not registered_tiers:
        adapter_dir = Path(config.adapter_dir)
        # No separate payload-sniff check: any main tier carrying either
        # venue's payload also carries a registry file (both commit
        # primitives write the registry last, as the commit signal), so
        # _has_adapter_registry already corroborates disk content on its own.
        disk_has_content = any(_has_adapter_registry(adapter_dir, t) for t in MAIN_TIERS) or any(
            True for _ in iter_interim_dirs(adapter_dir)
        )
        if disk_has_content:
            raise RuntimeError(
                "active-store migration: live store registered 0 tiers but "
                "on-disk content exists — registries failed to load; refusing "
                "to complete a no-op migration"
            )
        # Legitimately empty store (fresh install, no keys, no on-disk content):
        # fall through — all_tiers_done([]) is vacuously True, state cleared.

    # This migration retrains and rewrites every main tier's weight slot and
    # registry (below) -- content-PRESERVING, not content-replacing, so a
    # pending consolidation event's ledger cannot be discarded to make room
    # for it (that rule is reserved for operations that replace the content
    # the event derived against, e.g. a snapshot-bundle restore or a
    # base-swap rollback).  Refuse instead, path-only, before ``save_state``
    # below so a first-time mode-switch trigger never creates the migration
    # state file over a tree that is not yet coherent.
    from paramem.training import stage_ledger as _sl

    _sl_state_dir = _sl.data_state_dir(config.paths.data)
    if _sl.read_ledger(_sl_state_dir) is not None:
        raise RuntimeError(
            "active-store migration: a pending consolidation event must complete "
            "before the store migration can run — POST /consolidate resumes it, "
            "or the schedule will on its own next dispatch"
        )

    save_state(config.adapter_dir, state)  # ensure file exists at start

    for name in registered_tiers:
        if name in state.completed_tiers:
            logger.info("active_store_migration: store %s already complete, skipping", name)
            continue
        try:
            if state.target_mode == "train":
                _migrate_tier_simulate_to_train(loop, config, name)
            else:
                _migrate_tier_train_to_simulate(loop, config, name)
            state.completed_tiers.append(name)
            state.failed_tiers.pop(name, None)
            save_state(config.adapter_dir, state)
            logger.info("active_store_migration: store %s migrated successfully", name)
        except _TierSkipped as exc:
            logger.info("active_store_migration: store %s skipped: %s", name, exc)
            state.completed_tiers.append(name)
            save_state(config.adapter_dir, state)
        except Exception as exc:  # noqa: BLE001 — top-level boundary
            logger.exception("active_store_migration: store %s failed", name)
            state.failed_tiers[name] = str(exc)
            save_state(config.adapter_dir, state)
            # Continue to remaining stores — operator can re-trigger to retry

    if state.all_tiers_done(registered_tiers):
        clear_state(config.adapter_dir)
        logger.info(
            "active_store_migration: %s complete; state file removed",
            state.direction,
        )

    return state


# ---------------------------------------------------------------------------
# Per-tier implementations
# ---------------------------------------------------------------------------


def _delete_weight_slots(slot_root: Path) -> int:
    """Delete adapter weight-slot subdirectories under *slot_root*.

    A weight slot is a subdirectory containing ``adapter_model.safetensors``
    or ``adapter_config.json``. A simulate-payload slot (its ``graph.json``
    lives inside its OWN timestamped subdirectory, never at *slot_root*'s
    top level — nothing writes a tier-root ``graph.json`` any more) never
    matches this predicate and is preserved, as is ``indexed_key_registry.json``
    at *slot_root*'s top level.
    Returns the number of slots removed.
    """
    deleted = 0
    if slot_root.exists():
        for child in list(slot_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "adapter_model.safetensors").exists() or (
                child / "adapter_config.json"
            ).exists():
                shutil.rmtree(child)
                deleted += 1
    return deleted


def _delete_orphaned_simulate_slots(slot_root: Path, *, keep: Path) -> int:
    """Delete simulate-payload slot subdirectories under *slot_root* other
    than *keep* — the simulate→train crash-resume cleanup mirror of
    :func:`_delete_weight_slots`.

    On crash between the train-slot commit and the old simulate slot's
    delete (:func:`_migrate_tier_simulate_to_train`'s step 7→8), a resumed
    retry's ``find_live_slot`` binds the already-committed TRAIN slot —
    the stale simulate slot is invisible to that resolution but still on
    disk as a sibling directory. This sweeps it, mirroring
    :func:`_migrate_tier_train_to_simulate`'s own resume behaviour (which
    always runs its weight-slot cleanup after the already-written branch,
    never short-circuits without it).

    A slot is identified by its OWN manifest's ``payload.kind`` (venue-blind
    resolution, never a filename sniff) so this only ever removes a
    simulate-kind slot, never *keep* or an unrelated directory. A
    subdirectory whose ``meta.json`` is missing or unreadable is left
    alone — this is a best-effort orphan sweep, not
    ``cleanup_partial_slots``'s scratch-removal contract.

    Args:
        slot_root: Directory holding the tier's timestamped slots.
        keep: The slot to never delete (the tier's current bound slot).

    Returns:
        The number of slots removed.
    """
    from paramem.adapters.manifest import ManifestNotFoundError, ManifestSchemaError, read_manifest
    from paramem.memory.interim_adapter import INTERIM_DIR_PREFIX

    deleted = 0
    if not slot_root.exists():
        return deleted
    for child in list(slot_root.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child == keep:
            continue
        if child.name.startswith(INTERIM_DIR_PREFIX):
            # A main tier's slot root (e.g. <adapter_dir>/episodic/) holds
            # sibling interim_<stamp>/ containers, not written slots of THIS
            # tier — they are owned by find_live_slot + the boot-time
            # keyless-tier sweep, never by this orphan cleanup (mirrors the
            # same skip in integrity.py's partial-slot sweep).
            continue
        try:
            manifest = read_manifest(child)
        except (ManifestNotFoundError, ManifestSchemaError):
            continue
        if manifest.payload.kind == "simulate":
            shutil.rmtree(child)
            deleted += 1
    return deleted


def _migrate_tier_train_to_simulate(
    loop: "ConsolidationLoop", config: "ServerConfig", name: str
) -> None:
    """Switch a store from train to simulate by writing a graph slot and dropping weights.

    Handles both main tiers (``"episodic"``, ``"semantic"``, ``"procedural"``)
    and interim adapters (``"episodic_interim_<stamp>"``).  The on-disk tier
    root is resolved via :func:`adapter_slot_root_for_name` so the correct
    hierarchy is used for each store.

    Both venues write into the same tier root, each into its own timestamped
    slot (:func:`~paramem.adapters.slot.write_slot`) — the "active store"
    distinction is which payload kind the tier's BOUND slot's manifest
    declares, resolved by :func:`~paramem.adapters.manifest.find_live_slot`
    against :func:`~paramem.adapters.manifest.tier_registry_sha256`, never a
    flat filename check at the tier root.

    The migration to simulate:

    1. When the tier's currently-bound slot is already a simulate payload
       (a resumed retry that written but crashed before deleting the train
       slot(s) below), skip straight to step 3 — nothing to reconstruct or
       rewrite.
    2. Otherwise, reconstruct the graph from weights
       (:func:`~paramem.graph.reconstruct.reconstruct_graph`), verify it
       carries every active registry key (sanity check, BEFORE anything is
       written to disk — a failure here leaves the tree untouched, so there
       is nothing to roll back), then write it into a fresh simulate slot
       through the shared envelope
       (:func:`~paramem.adapters.manifest.graph_payload_manifest` +
       :func:`~paramem.adapters.slot.write_slot`) stamped with the tier's
       current registry hash — the same hash the just-written slot needs to
       bind on the next boot/reload.
    3. Delete all timestamped adapter weight-slot subdirectories under the
       tier root (directories containing ``adapter_model.safetensors`` or
       ``adapter_config.json``) — the newly-written simulate slot carries
       neither filename, so this never touches it.  The tier root, the
       simulate slot, and the tier-root registry/bookkeeping files are
       preserved — only the train payload is removed.

    Raises:
        _TierSkipped: When there are no active registry keys for this store.
        RuntimeError: When the reconstructed graph is missing an active key
            (sanity check) — raised BEFORE anything is written, so nothing is
            rolled back.
    """
    from paramem.adapters.manifest import find_live_slot, read_manifest, tier_registry_sha256
    from paramem.memory.interim_adapter import adapter_slot_root_for_name
    from paramem.memory.persistence import iter_entries

    active_keys = loop.store.active_keys_in_tier(name)
    slot_root = adapter_slot_root_for_name(Path(config.adapter_dir), name)

    if not active_keys:
        # Empty tier: nothing to write, but DELETE any stale weight slots so
        # the tier is cleanly simulate (no weights).  Critical for a
        # base-swap: an undeleted OLD-model slot survives both Phase A and Phase B
        # (each skips empty tiers), and the next boot/reload then reports a spurious
        # ``fingerprint_mismatch`` of that old-model slot against the NEW model
        # instead of a clean 0-key tier.  For a same-model mode-switch this is also
        # correct — an empty simulate tier should carry no weight slots.
        deleted = _delete_weight_slots(slot_root)
        raise _TierSkipped(
            f"no active registry keys for store {name}; deleted {deleted} stale weight slot(s)"
        )

    live_digest = tier_registry_sha256(slot_root)
    bound_slot = find_live_slot(slot_root, live_digest)
    already_written = (
        bound_slot is not None and read_manifest(bound_slot).payload.kind == "simulate"
    )

    if not already_written:
        from paramem.adapters.manifest import graph_payload_manifest
        from paramem.adapters.slot import payload_filename, write_slot
        from paramem.graph.reconstruct import ReconstructionError, reconstruct_graph
        from paramem.memory.persistence import _IK_KEY_ATTR, save_memory_to_disk

        try:
            result = reconstruct_graph(loop, tier=name, strict=True)
        except ReconstructionError as exc:
            raise RuntimeError(
                f"train_to_simulate store {name}: weight reconstruction failed: {exc}"
            ) from exc
        graph = result.graph
        for subject, obj, eid, data in graph.edges(keys=True, data=True):
            ik_key = data.get(_IK_KEY_ATTR)
            if ik_key is None:
                continue
            data["speaker_id"] = loop.store.bookkeeping_for_key(ik_key)["speaker_id"]

        # Sanity check BEFORE writing — nothing is written yet, so a failure
        # here leaves the tree exactly as it was; there is no write to roll
        # back.
        graph_keys = {e["key"] for e in iter_entries(graph)}
        missing = [k for k in active_keys if k not in graph_keys]
        if missing:
            raise RuntimeError(
                f"train_to_simulate store {name}: sanity check failed — "
                f"{len(missing)} key(s) missing from the reconstructed graph: "
                f"{missing[:5]!r}{'...' if len(missing) > 5 else ''}"
            )

        manifest = graph_payload_manifest(
            name=name,
            key_count=len(active_keys),
            registry_sha256=live_digest,
            # This path never stamps a cadence window (same as the
            # simulate_to_train direction's commit_tier_slot call below) —
            # window_stamp is provenance-only, read by nothing else.
            window_stamp="",
        )

        def _write_graph_payload(pending_slot: Path, _graph=graph) -> None:
            save_memory_to_disk(_graph, pending_slot / payload_filename("simulate"))

        bound_slot = write_slot(slot_root, manifest=manifest, write_payload=_write_graph_payload)
        logger.info(
            "train_to_simulate store %s: written simulate slot -> %s (%d keys)",
            name,
            bound_slot,
            len(active_keys),
        )

    # Drop adapter weight-slot subdirectories from the tier root — the
    # freshly-written simulate slot carries neither adapter_model.safetensors
    # nor adapter_config.json, so it is never among them.
    deleted_slots = _delete_weight_slots(slot_root)
    logger.info(
        "active_store_migration: store %s switched to simulate;"
        " %d keys retained in simulate slot %s; deleted %d weight slot(s) from %s",
        name,
        len(active_keys),
        bound_slot,
        deleted_slots,
        slot_root,
    )


def _migrate_tier_simulate_to_train(
    loop: "ConsolidationLoop", config: "ServerConfig", name: str
) -> None:
    """Read the bound simulate-slot's graph.json → train into ``<name>`` adapter →
    probe at ``loop.config.recall_sanity_threshold`` → on pass, persist slot + delete source slot.

    Handles both main tiers (``"episodic"``, ``"semantic"``,
    ``"procedural"``) and interim adapters
    (``"episodic_interim_<stamp>"``).  The on-disk tier root and the LoRA
    config are both resolved by name — interim stores use the episodic config
    and the path under ``<adapter_dir>/episodic/interim_<stamp>/``.

    Caller must hold the GPU lock — training and the recall probe both
    drive the model forward and would race STT/TTS otherwise.

    The simulate-mode store holds entries in the tier's bound slot's
    ``graph.json``, resolved the same way the train venue resolves its own
    bound slot: :func:`~paramem.adapters.manifest.find_live_slot` against
    :func:`~paramem.adapters.manifest.tier_registry_sha256`.

    Sequence:

    1. Resolve the tier's bound slot; skip (``_TierSkipped``) when there is
       none, or when its manifest already declares a ``"train"`` payload (a
       resumed retry that committed the train slot but crashed before
       deleting the source simulate slot below — this arm sweeps any
       orphaned simulate slot(s) still on disk, via
       :func:`_delete_orphaned_simulate_slots`, before skipping, mirroring
       :func:`_migrate_tier_train_to_simulate`'s own resume cleanup).  Load
       the bound slot's ``graph.json``; extract entry dicts via
       ``iter_entries``.
    2. Hot-load into ``loop.store`` + register keys into the per-store
       registry inside ``loop.store`` with ``adapter_id=name`` so the recall
       probe can find them.  The store entry itself is content-only
       (``{key, subject, predicate, object}``) — attribution lives only in
       bookkeeping.  Each registration is paired with a bookkeeping record
       (``store.set_bookkeeping``), sourced from the live store (which
       already carries the on-disk ``key_metadata.json`` it was hydrated
       from), then the source graph entry's own ``speaker_id`` — see the
       hot-load loop for the priority order.  Neither source having a value
       raises (no-unattributed-keys invariant; see the hot-load loop).
    3. Reset the adapter to LoRA-zero
       (``delete_adapter`` + ``create_adapter`` from the resolved config),
       then ``switch_adapter`` so training writes into this adapter.
    4-5. ``loop._train_tier_adapter(entries, ...)`` — the SAME shared
       training funnel every production training path uses
       (``paramem.training.consolidation.ConsolidationLoop._train_tier_adapter``):
       formats entries, builds the HF dataset, derives the per-fold training
       budget from ``len(entries)``, wires the recall-early-stop callback,
       and calls ``train_adapter``. Budget and callback wiring are inherited
       here rather than duplicated.  Training leaves the staged weights
       resident under ``STAGING_ADAPTER``; *name* stays at the LoRA-zero
       state Step 3 put it in until the promote below.
    6. Probe the staged weights (``loop._probe_recall(STAGING_ADAPTER,
       entries)``) at ``loop.config.recall_sanity_threshold`` (the unified
       recall gate knob), inside :func:`~paramem.training.trainer.staged_weights`.
       On pass, :func:`~paramem.training.trainer.promote_staging_adapter`
       copies the staged weights into *name*; on refusal, raises
       :class:`~paramem.training.consolidation.RecallGateRejected` and *name*
       is never touched — it stays exactly at Step 3's LoRA-zero state, so
       there is nothing to roll back.
    7. On pass: :func:`~paramem.memory.persistence.commit_tier_slot` (train
       mode) writes the adapter weight slot under the resolved tier root and
       flushes ``indexed_key_registry.json`` (carrying the unified simhash map)
       as the commit signal — the same per-tier commit primitive every other
       durable tier write (interim slots, main-tier folds, the trial tree)
       uses.  Delete the source simulate slot directory (not merely its
       ``graph.json`` file — the whole timestamped slot, meta.json included)
       — guarded on ``source_slot.exists()``: ``commit_tier_slot``'s own
       ``prune_old_slots`` call may already have retired it as a prior slot
       of the same tier root, so a fully successful migration must not fail
       on an already-gone source slot.
    """
    from paramem.adapters.manifest import find_live_slot, read_manifest, tier_registry_sha256
    from paramem.adapters.slot import payload_filename
    from paramem.memory.entry import build_registry as _build_reg
    from paramem.memory.entry import content_only_entry
    from paramem.memory.interim_adapter import adapter_slot_root_for_name
    from paramem.memory.persistence import commit_tier_slot, iter_entries, load_memory_from_disk
    from paramem.models.loader import create_adapter, switch_adapter

    # Resolve the tier's bound slot the same way the train venue does.
    slot_root = adapter_slot_root_for_name(Path(config.adapter_dir), name)
    source_slot = find_live_slot(slot_root, tier_registry_sha256(slot_root))
    if source_slot is None:
        raise _TierSkipped(f"no bound slot under {slot_root}")
    source_manifest = read_manifest(source_slot)
    if source_manifest.payload.kind != "simulate":
        # Resume after a crash between the train-slot commit (step 7) and
        # the source-slot delete (step 8): the tier's bound slot is already
        # train — sweep any orphaned simulate slot(s) left behind by the
        # interrupted step 8 before skipping, so a resumed migration still
        # converges on a clean tree instead of leaking the old payload
        # forever.
        _delete_orphaned_simulate_slots(slot_root, keep=source_slot)
        raise _TierSkipped(
            f"bound slot {source_slot} already carries a "
            f"{source_manifest.payload.kind!r} payload — nothing to migrate"
        )
    source_graph = source_slot / payload_filename("simulate")

    graph = load_memory_from_disk(source_graph)
    entries = list(iter_entries(graph))
    if not entries:
        raise _TierSkipped(f"empty graph.json at {source_graph}")

    # Registry-authoritative filter: graph.json can transiently be a
    # SUPERSET of the tier's ACTIVE keys — a stale edge still on disk after
    # the registry has withheld (or never knew) the key. Producers of this
    # shape are a torn restore (registry and graph.json restored out of
    # order — see ``restore_bundle``'s registry-last rule), a torn simulate
    # write (``commit_tier_slot`` crashes between writing graph.json and the
    # registry flush that follows it), or an ordinary operator erase: the
    # forget door no longer erases graph content at all, it stale-marks, and
    # a withheld key's graph.json edge must not be hot-loaded and retrained
    # here — ``MemoryStore.put``'s ``register=True`` default would
    # re-register it as active, resurrecting the erasure. The registry is
    # the lifecycle authority regardless of producer, so when it holds any
    # active key, filter entries down to the keys it still holds ACTIVE.
    # A side effect: ``replace_simhashes_in_tier`` (below, step 2) refuses a
    # ``new_simhashes`` map naming a withheld id — with this filter in
    # place, ``entries`` (and so the map it builds) never names one, so
    # that refusal path is unreachable from this caller.
    active_keys = set(loop.store.registry(name).list_active())
    if active_keys:
        filtered_entries = [e for e in entries if e["key"] in active_keys]
        dropped = len(entries) - len(filtered_entries)
        if dropped:
            logger.warning(
                "simulate_to_train store %s: dropped %d graph entr%s not active in the "
                "tier registry (withheld or orphaned graph content, e.g. a torn restore, "
                "a torn simulate write, or an operator erase)",
                name,
                dropped,
                "y" if dropped == 1 else "ies",
            )
        entries = filtered_entries
        if not entries:
            raise _TierSkipped(
                f"all graph entries at {source_graph} filtered out — "
                f"none are active in the {name} tier registry"
            )
    else:
        # No ACTIVE key in the in-memory registry is ambiguous UNLESS the
        # on-disk file's presence disambiguates it:
        #
        # * Registry file PRESENT is authoritative whether it holds withheld
        #   markers or nothing at all — the tier has no active key, so its
        #   graph content is not migrated and the tier is skipped. The
        #   producers of this shape are an operator erase that withheld the
        #   tier's last active key, and a rebuild that published a tier with
        #   none.
        # * Registry file ABSENT is the torn weight-before-registry-write case
        #   (shared by commit_tier_slot and the fold's write_tier_slot +
        #   publish_tier_registry — the registry write never landed) —
        #   indistinguishable from a transient unmounted tier and can never
        #   prove orphanhood, so all entries are kept and a WARNING is
        #   logged instead (unchanged from before this guard was added).
        registry_file = slot_root / "indexed_key_registry.json"
        if registry_file.exists():
            raise _TierSkipped(
                f"tier registry at {registry_file} is present with no active key — "
                f"treating {len(entries)} graph entr{'y' if len(entries) == 1 else 'ies'} "
                f"at {source_graph} as already-erased, not migrating"
            )
        logger.warning(
            "simulate_to_train store %s: tier registry file is absent at %s — migrating "
            "all %d graph entries unfiltered (an absent registry cannot distinguish a "
            "torn weight-before-registry commit write from a transient unmounted tier)",
            name,
            registry_file,
            len(entries),
        )

    # loop._tier_adapter_config is the one rule home for the
    # interim-is-episodic-shaped fallback; its KeyError (name is neither a
    # resident main tier nor an interim adapter name — e.g. a main tier the
    # operator has disabled) converts to this module's own skip-and-continue
    # convention.
    try:
        tier_config = loop._tier_adapter_config(name)
    except KeyError:
        raise _TierSkipped(f"store {name}: not in loop.tier_adapters") from None

    # Bookkeeping source for the hot-load loop below, resolved once (not
    # per key).  ``iter_entries`` only carries {key, subject, predicate,
    # object, speaker_id} — no relation_type/reinforcement_count/timestamps
    # — so a registered key needs its provenance row from elsewhere.  The
    # base-swap worker already hydrates the loop store via
    # ``load_bookkeeping_from_disk`` before ``migrate()`` runs (see the
    # ordinary mode-switch path, which reuses ``loop`` — the live singleton,
    # whose store already carries bookkeeping loaded at boot), so
    # ``loop.store.bookkeeping_for_key(key)`` is the row source for both
    # venues; there is no separate on-disk read here.

    # Step 2: hot-load into the loop's memory store so the recall probe
    # (which reads from loop.store) can find the keys.  Mirrors the
    # seed_<tier>_cache methods in consolidation.py.
    loop.store.replace_simhashes_in_tier(name, _build_reg(entries))
    for kp in entries:
        key = kp["key"]
        # Store-entry shape is content-only ({key, subject, predicate,
        # object}) throughout this migration, matching the shape every other
        # store.put site in the fold writes (the shared
        # paramem.memory.entry.content_only_entry projection) — a fatter
        # entry here would leave a migrated key with residual
        # speaker_id/relation_type fields that /debug/dump would leak.
        # Attribution is carried only in the bookkeeping record set below.
        # iter_entries always yields all five fields (persistence.py), so
        # the projection applies directly without a defaulting pass.
        loop.store.put(name, key, content_only_entry(kp))

        # Pair the registration with a bookkeeping record — an active key
        # with no provenance row trips the main-tiers fold's
        # registry_bookkeeping_divergence integrity gate on the next full
        # consolidation.  Priority: an already-loaded live-store record
        # (the sole reader of on-disk bookkeeping now — see the module note
        # above), then the source graph entry's own speaker_id
        # (``iter_entries`` carries it — the same quantity
        # ``build_tier_graph_from_store`` wrote from bookkeeping when the
        # graph was produced, so it is in-hand attribution, not a guess).
        # Neither source has a value only against a store predating the
        # bookkeeping speaker invariant — set_bookkeeping now raises
        # (via paramem.memory.bookkeeping.bookkeeping_row) rather than
        # minting an unattributed row.  Accepted: a fresh store never
        # reaches this, and the raise is operator-visible on this base-swap
        # migration path.
        bk = loop.store.bookkeeping_for_key(key) or {}
        bk_speaker_id = bk.get("speaker_id") or kp.get("speaker_id", "")
        loop.store.set_bookkeeping(
            key,
            speaker_id=bk_speaker_id,
            relation_type=bk.get("relation_type", "unknown"),
            reinforcement_count=bk.get("reinforcement_count", 1),
            last_reinforced_cycle=bk.get("last_reinforced_cycle", 0),
            last_seen=bk.get("last_seen", ""),
            first_seen=bk.get("first_seen", ""),
            promoted=bk.get("promoted", False),
        )

    # Step 3: reset adapter to LoRA-zero (delete + recreate) -- the explicit
    # cold-rebuild semantics this migration is documented to use
    # (architecture.md's base-swap migration decision). This LoRA-zero state
    # is measured as "cold" by _train_tier_adapter below, which
    # unconditionally donor-seeds it (paramem.training.donor) before
    # training -- migration no longer starts from a bare LoRA-zero fact-free
    # state, it starts from the donor's task-skilled weights. On a
    # base-model swap (Phase B retraining onto the new base) the donor
    # checkpoint's recorded base_model_id no longer matches, so
    # donor_checkpoint_valid rejects it and the funnel builds a fresh donor
    # for the new base inline before seeding -- an intended consequence of
    # donor seeding being unconditional, not a special case here.
    if name in loop.model.peft_config:
        loop.model.delete_adapter(name)
    create_adapter(loop.model, tier_config, name)
    switch_adapter(loop.model, name)

    # Steps 4-5: format + dataset + budget derivation + recall callback +
    # train_adapter, all via the single shared funnel. Output dir under a
    # migration-scoped subdir so checkpoint debris doesn't pollute the main
    # slot layout.
    _migrate_output_dir = Path(config.adapter_dir) / "active_store_migration" / name
    # recall_state is intentionally unused here — the migration path uses its
    # own staged-weights probe gate below; no recall-gated registration
    # needed.
    _migrate_metrics, _recall_state = loop._train_tier_adapter(
        entries,
        adapter_name=name,
        adapter_config=tier_config,
        training_config=loop.training_config,
        output_dir=_migrate_output_dir,
        run_name=f"migrate-simulate-to-train-{name}",
        phase_name=f"migrate-{name}",
    )
    if _migrate_metrics is None:
        raise _TierSkipped(f"_train_tier_adapter produced no examples for store {name}")
    if _migrate_metrics.get("aborted"):
        raise _TierSkipped(f"aborted mid-migration for {name}")

    # Step 6: probe the staged weights and gate at the configured sanity
    # threshold, BEFORE promotion — uncapped by construction (the gate
    # primitive probes the full entries list; no sampling cap).  ``entries``
    # is unique per key (one graph edge per ik_key — see the module's own
    # key-assignment invariant), so RecallProbe.rate's distinct-key
    # denominator here is the same as len(entries).
    # Deliberate: the uncapped probe applies to ALL simulate→train migrations
    # (both the ordinary mode-switch path and Phase B of a base-swap).  Full
    # coverage is strictly safer than a sampled gate, matching the callback.
    # Cost: O(n) inference calls per store — budget accordingly for large stores.
    _migration_threshold = loop.config.recall_sanity_threshold
    with staged_weights(loop.model, fallback_adapter=name):
        probe = loop._probe_recall(STAGING_ADAPTER, entries)
        if probe.rate < _migration_threshold:
            # No rollback needed: the staged weights are disposed of by
            # staged_weights' own finally, and production never left the
            # LoRA-zero state Step 3 put it in.
            raise RecallGateRejected(
                f"simulate_to_train store {name} recall {probe.rate:.3f} < "
                f"{_migration_threshold:.3f}",
                adapter_name=name,
                recall_rate=probe.rate,
                threshold=_migration_threshold,
                failed_keys=tuple(sorted({r["key"] for r in probe.failed})),
            )
        promote_staging_adapter(loop.model, name)

    # Step 7: commit through the one per-tier commit primitive — the same
    # atomic write/registry-last/failure-cleanup sequence every other durable
    # tier write (interim slots, main-tier folds, the trial-migration tree)
    # uses.  This also durably persists bookkeeping via the commit
    # primitive's own per-tier ``key_metadata.json`` write (ahead of the
    # registry flush), prunes old
    # slots via ``prune_old_slots`` (after the registry flush, keeping
    # ``loop._keep_prior_slots`` prior slots), writes the debug-gated weight
    # shadow (``on_main_adapters_saved``, a no-op when snapshots are off), and,
    # on any failure before the registry flush lands, removes the orphan slot
    # this call wrote — none of which the hand-rolled sequence this replaced
    # did.  A manifest-build failure now propagates (loud, not swallowed):
    # the slot would otherwise be unmountable on the next boot because
    # find_live_slot can never match a slot with no manifest hash.
    # ``stamp=""`` preserves this path's pre-existing behaviour of never
    # stamping a cadence window on the manifest (``window_stamp`` is
    # provenance-only, read by nothing else).
    _written_slot = commit_tier_slot(
        loop=loop,
        tier=name,
        adapter_name=name,
        stamp="",
        mode="train",
        all_keyed=entries,
        output_dir=Path(config.adapter_dir),
    )

    # Step 8: delete the source simulate slot (target is now authoritative +
    # probe-confirmed) — the whole timestamped directory, not merely its
    # graph.json file. Guarded: commit_tier_slot's own prune_old_slots call
    # (step 7, above) may already have retired this exact slot as a prior
    # slot of the SAME tier root — at training_keep_prior_slots=0 always,
    # and at the default (3) once more than 3 prior slots already exist —
    # so a fully successful migration must not fail on an already-gone
    # source slot; prune owns prior-slot retirement, this call is a no-op
    # when it already ran.
    if source_slot.exists():
        shutil.rmtree(source_slot)

    logger.info(
        "active_store_migration: store %s migrated to train; slot=%s, %d keys",
        name,
        _written_slot,
        len(entries),
    )
