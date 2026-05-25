"""Migrate keyed facts between train (LoRA weights) and simulate (graph.json) stores.

Triggered when the operator flips ``consolidation.mode`` in server.yaml.
The migration is per-store with a 1.0 recall gate before cleanup; the
source store stays authoritative until ALL stores have completed and the
state file is removed. On crash mid-migration, the state file persists
and the system retains the source mode on next boot.

State file location: ``<paths.adapters>/.active_store_migration.json``
(age-encrypted via ``write_infra_bytes`` when the daily identity is loaded).

Under the unified layout (2026-05-16), ``graph.json`` lives at
``<adapter_dir>/<tier>/graph.json`` in **both** train and simulate modes.
The distinction between modes is whether adapter weight slot subdirectories
(containing ``adapter_model.safetensors``) exist alongside the graph.

Two directions:

* ``simulate_to_train``: read ``<adapter_dir>/<name>/graph.json`` →
  train into ``<name>`` adapter → recall probe at threshold=1.0 → on
  pass, atomic-save the slot. On fail, leave the graph intact.

* ``train_to_simulate``: verify ``<adapter_dir>/<name>/graph.json`` exists
  and covers all active keys; reconstruct from weights if missing →
  delete all timestamped weight slot subdirs under the resolved slot root
  (graph.json and registries are preserved). On fail, remove any
  freshly-written graph.json and leave the adapter slots intact.

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

``_has_tier_graph`` walks subdirectories so interim simulate-mode graph.json
files under ``<adapter_dir>/<tier>/interim_<stamp>/`` are detected correctly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig
    from paramem.training.consolidation import ConsolidationLoop

logger = logging.getLogger(__name__)


_STATE_FILENAME = ".active_store_migration.json"
TIERS: tuple[str, ...] = ("episodic", "semantic", "procedural")


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


def _has_tier_graph(adapter_dir: Path, tier: str) -> bool:
    """Return True when a per-tier ``graph.json`` exists under the unified layout.

    Under the unified layout graph.json lives at
    ``<adapter_dir>/<tier>/graph.json`` for both simulate and train modes.
    In simulate mode, interim cycles write graph.json under
    ``<adapter_dir>/<tier>/interim_<stamp>/graph.json`` — those subdirectory
    files count as well.

    Walks the tier directory tree so both the main-slot file and any
    interim simulate-mode slot files are detected.
    """
    tier_root = Path(adapter_dir) / tier
    if not tier_root.is_dir():
        return False
    return any(tier_root.rglob("graph.json"))


def _has_adapter_registry(adapter_dir: Path, tier: str) -> bool:
    """Return True when a per-tier indexed_key_registry.json exists for *tier*.

    The canonical layout is ``<adapter_dir>/<tier>/indexed_key_registry.json``.
    This is the commit signal written last by ``_save_adapters`` (I5 ordering).
    """
    return (Path(adapter_dir) / tier / "indexed_key_registry.json").exists()


def detect_mode_switch(config: "ServerConfig") -> Optional[MigrationState]:
    """Detect if the active-store state diverges from ``config.consolidation.mode``.

    Detection logic:

    1. If a state file exists, return it (migration was started, possibly
       interrupted, must be resumed before inference is consistent).
    2. Otherwise compare disk contents to the operator's yaml ``mode``:

       * ``mode=train`` and graph.json present but adapter registry absent
         → ``simulate_to_train`` migration is needed.
       * ``mode=simulate`` and adapter registry present but graph.json absent
         → ``train_to_simulate`` migration is needed.

    Returns ``None`` when the active store is consistent with the mode
    (no migration needed).
    """
    existing = load_state(config.adapter_dir)
    if existing is not None:
        return existing

    target_mode = config.consolidation.mode
    if target_mode not in ("simulate", "train"):
        return None  # unsupported mode — let upstream complain

    simulate_present = any(_has_tier_graph(config.adapter_dir, t) for t in TIERS)
    adapter_present = any(_has_adapter_registry(config.adapter_dir, t) for t in TIERS)

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

    Returns the updated state.
    """
    from paramem.memory.interim_adapter import iter_interim_dirs

    registered_tiers = loop.store.tiers_with_registry()

    if not registered_tiers:
        adapter_dir = Path(config.adapter_dir)
        disk_has_content = (
            any(_has_adapter_registry(adapter_dir, t) for t in TIERS)
            or any(_has_tier_graph(adapter_dir, t) for t in TIERS)
            or any(True for _ in iter_interim_dirs(adapter_dir))
        )
        if disk_has_content:
            raise RuntimeError(
                "active-store migration: live store registered 0 tiers but "
                "on-disk content exists — registries failed to load; refusing "
                "to complete a no-op migration"
            )
        # Legitimately empty store (fresh install, no keys, no on-disk content):
        # fall through — all_tiers_done([]) is vacuously True, state cleared.

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


def _migrate_tier_train_to_simulate(
    loop: "ConsolidationLoop", config: "ServerConfig", name: str
) -> None:
    """Switch a store from train to simulate by removing adapter weight slots.

    Handles both main tiers (``"episodic"``, ``"semantic"``, ``"procedural"``)
    and interim adapters (``"episodic_interim_<stamp>"``).  The on-disk slot
    root is resolved via :func:`adapter_slot_root_for_name` so the correct
    hierarchy is used for each store.

    Under the unified layout (2026-05-16), ``graph.json`` lives at the slot
    root in **both** train and simulate modes.  The "active store" distinction
    is just "are adapter weight slots present alongside the graph?":

    * train: timestamped weight slot dirs + graph.json + registries
    * simulate: graph.json + registries only (no weight slots)

    The migration to simulate therefore reduces to:

    1. Verify ``<slot_root>/graph.json`` exists and carries every active
       registry key (sanity check).  If missing (legacy deployment that ran
       before graph.json was universal), fall back to weight reconstruction
       once to materialise it.
    2. Delete all timestamped adapter weight slot subdirectories under
       *slot_root* (directories containing ``adapter_model.safetensors`` or
       ``adapter_config.json``).  The top-level slot directory, graph.json,
       and registries are preserved — only the weight payload is removed.

    Raises:
        _TierSkipped: When there are no active registry keys for this store.
        RuntimeError: When the post-step sanity check fails (graph.json
            missing keys after reconstruction).
    """
    from paramem.memory.interim_adapter import adapter_slot_root_for_name
    from paramem.memory.persistence import (
        iter_entries,
        load_memory_from_disk,
    )

    if not loop.store.replay_enabled:
        raise _TierSkipped(f"replay disabled on loop; store {name} skipped")
    active_keys = loop.store.active_keys_in_tier(name)
    if not active_keys:
        raise _TierSkipped(f"no active registry keys for store {name}")

    slot_root = adapter_slot_root_for_name(Path(config.adapter_dir), name)
    target_graph = slot_root / "graph.json"

    # Step 1: ensure graph.json is current.  Post-architecture cycles
    # write it on every consolidation, so this is the fast path.  On
    # legacy deployments (trained before graph.json was universal) we
    # reconstruct it from weights once.
    needs_reconstruction = not target_graph.exists()
    if not needs_reconstruction:
        loaded = load_memory_from_disk(target_graph)
        graph_keys = {q["key"] for q in iter_entries(loaded)}
        if not all(k in graph_keys for k in active_keys):
            needs_reconstruction = True
            logger.info(
                "train_to_simulate store %s: graph.json present but missing keys — "
                "reconstructing from weights",
                name,
            )

    if needs_reconstruction:
        from paramem.graph.reconstruct import ReconstructionError, reconstruct_graph
        from paramem.memory.persistence import (
            _IK_KEY_ATTR,
            save_memory_to_disk,
        )

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
            cache_entry = loop.store.get(ik_key) or {}
            data["speaker_id"] = cache_entry.get("speaker_id", "")
            data["first_seen_cycle"] = cache_entry.get("first_seen_cycle", 0)
        slot_root.mkdir(parents=True, exist_ok=True)
        save_memory_to_disk(graph, target_graph)

        # Sanity-check after reconstruction.
        loaded = load_memory_from_disk(target_graph)
        graph_keys = {q["key"] for q in iter_entries(loaded)}
        missing = [k for k in active_keys if k not in graph_keys]
        if missing:
            target_graph.unlink(missing_ok=True)
            raise RuntimeError(
                f"train_to_simulate store {name}: sanity check failed — "
                f"{len(missing)} key(s) missing after reconstruction: "
                f"{missing[:5]!r}{'...' if len(missing) > 5 else ''}; "
                f"rolled back simulate-store write"
            )

    # Step 2: drop adapter weight slot subdirectories from the slot root.
    # Weight slots are subdirectories that contain adapter_model.safetensors
    # or adapter_config.json.  graph.json, simhash_registry.json, and
    # indexed_key_registry.json live at the top of slot_root and are preserved.
    deleted_slots = 0
    if slot_root.exists():
        for child in list(slot_root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "adapter_model.safetensors").exists() or (
                child / "adapter_config.json"
            ).exists():
                shutil.rmtree(child)
                deleted_slots += 1
    logger.info(
        "active_store_migration: store %s switched to simulate;"
        " %d keys retained in graph.json; deleted %d weight slot(s) from %s",
        name,
        len(active_keys),
        deleted_slots,
        slot_root,
    )


def _tier_adapter_config(loop: "ConsolidationLoop", name: str):
    """Resolve the ``AdapterConfig`` for a store name from the loop.

    Interim adapter names (``"episodic_interim_<stamp>"``) are mapped to the
    episodic config because interim adapters are always topology-compatible
    with the main episodic adapter (same rank, alpha, and target modules).

    Raises ``_TierSkipped`` when the resolved tier is configured-out
    (e.g. ``procedural_config is None`` after the operator disabled procedural).
    """
    from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

    lookup = "episodic" if name.startswith(INTERIM_NAME_PREFIX) else name
    cfg_map = {
        "episodic": loop.episodic_config,
        "semantic": loop.semantic_config,
        "procedural": loop.procedural_config,
    }
    tier_config = cfg_map.get(lookup)
    if tier_config is None:
        raise _TierSkipped(f"store {name} (lookup={lookup!r}) not enabled (config is None)")
    return tier_config


def _migrate_tier_simulate_to_train(
    loop: "ConsolidationLoop", config: "ServerConfig", name: str
) -> None:
    """Read simulate-store graph.json → train into ``<name>`` adapter →
    probe at threshold=1.0 → on pass, persist slot + delete source graph.

    Handles both main tiers (``"episodic"``, ``"semantic"``,
    ``"procedural"``) and interim adapters
    (``"episodic_interim_<stamp>"``).  The on-disk slot root and the LoRA
    config are both resolved by name — interim stores use the episodic config
    and the path under ``<adapter_dir>/episodic/interim_<stamp>/``.

    Caller must hold the GPU lock — training and the recall probe both
    drive the model forward and would race STT/TTS otherwise.

    The simulate-mode store holds entries in ``graph.json``.

    Sequence:

    1. Load source graph from the resolved slot root ``graph.json``; extract
       entry dicts via ``iter_entries``.
    2. Hot-load into ``loop.store`` + register keys into the per-store
       registry inside ``loop.store`` with ``adapter_id=name`` so the recall
       probe can find them.
    3. Reset the adapter to LoRA-zero
       (``delete_adapter`` + ``create_adapter`` from the resolved config),
       then ``switch_adapter`` so training writes into this adapter.
    4. ``format_entry_training`` + ``_indexed_dataset`` to build the
       HF dataset; gradient checkpointing toggled around the format call
       to mirror the existing post_session_train pattern.
    5. ``train_adapter`` with the resolved adapter config and the loop's
       configured num_epochs.
    6. Recall probe via ``loop._run_recall_sanity_probe(name, entries)``
       at threshold 1.0 — stricter than the 0.95 in-process default.
    7. On pass: ``atomic_save_adapter`` writes the slot under the resolved
       slot root, SimHash registry written to
       ``<slot_root>/simhash_registry.json``.
       Delete the source graph.json.
    8. On fail: reset the adapter back to LoRA-zero so failure does not
       leave half-trained weights resident, raise ``RuntimeError``.
    """
    from paramem.adapters.manifest import build_manifest_for
    from paramem.memory.entry import (
        build_registry as _build_reg,
    )
    from paramem.memory.entry import (
        format_entry_training as _format_training,
    )
    from paramem.memory.interim_adapter import adapter_slot_root_for_name
    from paramem.memory.persistence import iter_entries, load_memory_from_disk
    from paramem.models.loader import (
        atomic_save_adapter,
        create_adapter,
        switch_adapter,
    )
    from paramem.training.trainer import TrainingHooks
    from paramem.training.trainer import train_adapter as _train_adapter

    # Source graph is at the unified layout location resolved by name.
    slot_root = adapter_slot_root_for_name(Path(config.adapter_dir), name)
    source_graph = slot_root / "graph.json"
    if not source_graph.exists():
        raise _TierSkipped(f"no graph.json at {source_graph}")

    graph = load_memory_from_disk(source_graph)
    entries = list(iter_entries(graph))
    if not entries:
        raise _TierSkipped(f"empty graph.json at {source_graph}")

    tier_config = _tier_adapter_config(loop, name)

    # Step 2: hot-load into the loop's memory store so the recall probe
    # (which reads from loop.store) can find the keys.  Mirrors the
    # seed_<tier>_cache methods in consolidation.py.
    loop.store.replace_simhashes_in_tier(name, _build_reg(entries))
    for kp in entries:
        key = kp["key"]
        entry = loop._cache_entry(
            key=key,
            subject=kp.get("subject", ""),
            predicate=kp.get("predicate", ""),
            object=kp.get("object", ""),
            speaker_id=kp.get("speaker_id", ""),
            first_seen_cycle=kp.get("first_seen_cycle", 0),
            question=kp.get("question"),
            answer=kp.get("answer"),
        )
        loop.store.put(name, key, entry)

    # Step 3: reset adapter to LoRA-zero so training starts from a clean state.
    if name in loop.model.peft_config:
        loop.model.delete_adapter(name)
    loop.model = create_adapter(loop.model, tier_config, name)
    switch_adapter(loop.model, name)

    # Step 4: build training dataset. Disable gradient checkpointing for the
    # tokenizer call (matches the post_session_train pattern at
    # consolidation.py:2692-2696); re-enable before training.
    loop._disable_gradient_checkpointing()
    examples = _format_training(entries, loop.tokenizer, max_length=1024)
    if not examples:
        raise _TierSkipped(f"_format_training produced no examples for store {name}")
    dataset = loop._indexed_dataset(examples)
    loop._enable_gradient_checkpointing()
    training_config = loop._make_training_config(num_epochs=loop.training_config.num_epochs)

    # Step 5: train. Output dir under a migration-scoped subdir so checkpoint
    # debris doesn't pollute the main slot layout.
    recall_cb = loop._maybe_make_recall_callback(
        entries=entries,
        adapter_name=name,
        output_dir=Path(config.adapter_dir) / "active_store_migration" / name,
        phase_name=f"migrate-{name}",
    )
    _train_adapter(
        model=loop.model,
        tokenizer=loop.tokenizer,
        train_dataset=dataset,
        adapter_name=name,
        training_config=training_config,
        adapter_config=tier_config,
        wandb_config=loop.wandb_config,
        output_dir=Path(config.adapter_dir) / "active_store_migration" / name,
        run_name=f"migrate-simulate-to-train-{name}",
        thermal_policy=loop._thermal_policy,
        hooks=TrainingHooks(on_shutdown_check=lambda: loop.shutdown_requested),
        callbacks_extra=[recall_cb] if recall_cb is not None else None,
    )

    # Step 6: recall probe at threshold=1.0 (stricter than 0.95 default).
    # Pass max_probe=len(entries) so the gate is uncapped — all keys must pass,
    # not just a 100-entry sample.  The RecallEarlyStopCallback already probes
    # the full entry set (no cap), so this call is now consistent with it.
    # Deliberate: the uncapped probe applies to ALL simulate→train migrations
    # (both the ordinary mode-switch path and Phase B of a base-swap).  Full
    # coverage is strictly safer than a sampled gate, matching the callback.
    # Cost: O(n) inference calls per store — budget accordingly for large stores.
    recall = loop._run_recall_sanity_probe(name, entries, max_probe=len(entries))
    if recall < 1.0:
        # Rollback: reset adapter to LoRA-zero. The slot was not yet saved to
        # disk so there's nothing to delete on the filesystem side.
        loop.model.delete_adapter(name)
        loop.model = create_adapter(loop.model, tier_config, name)
        raise RuntimeError(
            f"simulate_to_train store {name} recall {recall:.3f} < 1.0; "
            f"rolled back trained adapter to LoRA-zero"
        )

    # Step 7a: atomic-save the slot. Manifest building can fail (e.g. base-model
    # hash unavailable); we save without manifest in that case so the weights
    # are durable even when the manifest sidecar isn't.
    # Bind the slot to the tier registry exactly as consolidation._save_adapters
    # does (I5): hash the registry bytes and pass them as
    # registry_sha256_override.  Without this the slot's meta.registry_sha256 is
    # empty, find_live_slot can never match it on the next boot/reload, and the
    # adapter silently fails to mount (recall then returns 0 keys → boot_degraded).
    _tier_reg = loop.store.registry(name)
    _reg_payload = _tier_reg.save_bytes() if _tier_reg is not None else None
    _reg_sha = hashlib.sha256(_reg_payload).hexdigest() if _reg_payload is not None else None

    fingerprint_cache = getattr(loop, "fingerprint_cache", None)
    try:
        manifest = build_manifest_for(
            loop.model,
            loop.tokenizer,
            name,
            registry_path=None,
            key_count=len(entries),
            base_model_hash_cache=fingerprint_cache,
            registry_sha256_override=_reg_sha,
            adapter_root=Path(config.adapter_dir),
        )
    except Exception:
        logger.warning(
            "active_store_migration: store %s manifest build failed — saving without manifest",
            name,
        )
        manifest = None
    slot_path = atomic_save_adapter(
        loop.model,
        slot_root,
        name,
        manifest=manifest,
    )

    # Step 7b: persist the per-store SimHash registry inside the slot root.
    # Mirrors training/consolidation.py::_save_adapters — without this file the
    # boot-time _load_simhash_registry returns an empty {} for this store, and
    # probe_entries / probe_key then treat every recalled key as untrained
    # (verify_confidence → 0.0 → low_confidence:0.000), so every personal query
    # silently abstains even though the adapter recalls correctly. The simhash
    # was already built in Step 2 (replace_simhashes_in_tier); persist it as
    # {key: int} shape that _load_simhash_registry / get_simhash expect.
    from paramem.memory.persistence import save_registry as _save_registry

    _simhash = loop.store.simhashes_in_tier(name)
    if _simhash:
        slot_root.mkdir(parents=True, exist_ok=True)
        _save_registry(
            _simhash,
            slot_root / "simhash_registry.json",
        )

    # Step 7c2: flush the exact registry bytes that were hashed into the manifest
    # so find_live_slot matches meta.registry_sha256 against the tier registry on
    # the next boot/reload (mirrors consolidation._save_adapters I5 step 8 — the
    # registry is the commit signal, written last).
    if _tier_reg is not None and _reg_payload is not None:
        slot_root.mkdir(parents=True, exist_ok=True)
        _tier_reg.save_from_bytes(
            _reg_payload, slot_root / "indexed_key_registry.json", consolidating=True
        )

    # Step 7d: delete source (target is now authoritative + probe-confirmed).
    source_graph.unlink()

    logger.info(
        "active_store_migration: store %s migrated to train; slot=%s, %d keys",
        name,
        slot_path,
        len(entries),
    )
