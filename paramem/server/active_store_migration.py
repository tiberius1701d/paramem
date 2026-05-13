"""Migrate keyed facts between train (LoRA weights) and simulate (graph.json) stores.

Triggered when the operator flips ``consolidation.mode`` in server.yaml.
The migration is per-tier with a 1.0 recall gate before cleanup; the
source store stays authoritative until ALL tiers have completed and the
state file is removed. On crash mid-migration, the state file persists
and the system falls back to the source mode on next boot.

State file location: ``<paths.adapters>/.active_store_migration.json``
(age-encrypted via ``write_infra_bytes`` when the daily identity is loaded).

Two directions:

* ``simulate_to_train``: read ``<simulate>/<tier>/graph.json`` →
  train into ``<tier>`` adapter → recall probe at threshold=1.0 → on
  pass, atomic-save the slot and delete the simulate-store graph.json. On
  fail, leave both stores intact.

* ``train_to_simulate``: reconstruct the tier graph from the adapter weights
  via ``reconstruct_graph(loop, tier=tier, strict=True)`` → decorate edges
  with speaker_id/first_seen_cycle from ``loop.indexed_key_cache`` →
  write to ``<simulate>/<tier>/graph.json`` → sanity-check every active key
  present in graph → on pass, delete the adapter tier dir.  On fail, remove
  the graph.json and leave the adapter slots intact.

Per-tier failures are recorded in the state file but do not abort the
remaining tiers — the operator can re-trigger to retry.

Interim adapters are NOT migrated: per the locked design decision
(see ``paramem/server/consolidation.py:170``), simulate mode has no
interim concept. ``episodic_interim_*`` slots are torn down by the
next full consolidation; the migration only operates on the three
main tiers (episodic, semantic, procedural).
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
    and not completed. Startup detection treats this as "fall back to
    ``source_mode`` for inference until completion".
    """

    direction: str  # "simulate_to_train" | "train_to_simulate"
    started_at: str  # iso8601
    source_mode: str  # "simulate" | "train" — fallback target on interrupt
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

    @property
    def all_tiers_done(self) -> bool:
        return all(t in self.completed_tiers for t in TIERS) and not self.failed_tiers


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


def _has_simulate_graph(simulate_dir: Path, tier: str) -> bool:
    """Return True when a simulate-mode ``graph.json`` exists for *tier*."""
    return (Path(simulate_dir) / tier / "graph.json").exists()


def _has_adapter_kp(adapter_dir: Path, tier: str) -> bool:
    """Is there a tier-level keyed_pairs.json under this adapter dir?

    The canonical layout (verified live) is ``<adapter_dir>/<tier>/keyed_pairs.json``
    at the **tier level** — written by ``_save_adapters._write_kp`` /
    ``_save_keyed_pairs_for_router``. Slot subdirectories
    (``<adapter_dir>/<tier>/<ts>/``) hold weight + manifest only.
    """
    return (Path(adapter_dir) / tier / "keyed_pairs.json").exists()


def detect_mode_switch(config: "ServerConfig") -> Optional[MigrationState]:
    """Detect if the active-store state diverges from ``config.consolidation.mode``.

    Detection logic:

    1. If a state file exists, return it (migration was started, possibly
       interrupted, must be resumed before inference is consistent).
    2. Otherwise compare disk contents to the operator's yaml ``mode``:

       * ``mode=train`` and simulate-store kp present and adapter slots
         absent → ``simulate_to_train`` migration is needed.
       * ``mode=simulate`` and adapter slots present and simulate-store
         kp absent → ``train_to_simulate`` migration is needed.

    Returns ``None`` when the active store is consistent with the mode
    (no migration needed).
    """
    existing = load_state(config.adapter_dir)
    if existing is not None:
        return existing

    target_mode = config.consolidation.mode
    if target_mode not in ("simulate", "train"):
        return None  # unsupported mode — let upstream complain

    simulate_present = any(_has_simulate_graph(config.simulate_dir, t) for t in TIERS)
    adapter_present = any(_has_adapter_kp(config.adapter_dir, t) for t in TIERS)

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

    Per-tier; each tier's success is persisted to the state file before
    moving to the next, so a crash mid-migration can resume from the
    last committed tier on the next call.

    Source store is preserved until ALL tiers have completed cleanly
    (state.all_tiers_done) — only then is the state file removed and
    the source-side artifacts deleted.

    Returns the updated state.
    """
    save_state(config.adapter_dir, state)  # ensure file exists at start

    for tier in TIERS:
        if tier in state.completed_tiers:
            logger.info("active_store_migration: tier %s already complete, skipping", tier)
            continue
        try:
            if state.target_mode == "train":
                _migrate_tier_simulate_to_train(loop, config, tier)
            else:
                _migrate_tier_train_to_simulate(loop, config, tier)
            state.completed_tiers.append(tier)
            state.failed_tiers.pop(tier, None)
            save_state(config.adapter_dir, state)
            logger.info("active_store_migration: tier %s migrated successfully", tier)
        except _TierSkipped as exc:
            logger.info("active_store_migration: tier %s skipped: %s", tier, exc)
            state.completed_tiers.append(tier)
            save_state(config.adapter_dir, state)
        except Exception as exc:  # noqa: BLE001 — top-level boundary
            logger.exception("active_store_migration: tier %s failed", tier)
            state.failed_tiers[tier] = str(exc)
            save_state(config.adapter_dir, state)
            # Continue to remaining tiers — operator can re-trigger to retry

    if state.all_tiers_done:
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
    loop: "ConsolidationLoop", config: "ServerConfig", tier: str
) -> None:
    """Reconstruct the tier's graph from adapter WEIGHTS and persist as simulate store.

    Source: adapter weights in ``<adapter_dir>/<tier>/``.
    Target: ``<simulate_dir>/<tier>/graph.json``.

    The previous implementation copied ``<adapter_dir>/<tier>/keyed_pairs.json``
    to ``<simulate_dir>/<tier>/keyed_pairs.json`` without ever touching the
    weights.  The invariant now enforced: every fact is extracted from the
    adapter's weights into the graph and persisted, so the simulate store is
    always a faithful reconstruction of what the model knows, not a copy of a
    sidecar that may have drifted.

    Steps:
    1. Verify there are active registry keys for this tier (else ``_TierSkipped``).
    2. ``reconstruct_graph(loop, tier=tier, strict=True)`` — probes every active
       key for this tier from the weights via probe_quad; raises
       ``ReconstructionError`` if any key fails (strict=True).
    3. Decorate every edge with ``speaker_id`` and ``first_seen_cycle`` from
       ``loop.indexed_key_cache[key]``.  Defaults to ``""`` / ``0`` when absent.
    4. ``save_simulate_graph(graph, <simulate_dir>/<tier>/graph.json)``.
    5. Sanity-check: every active registry key for this tier has an edge with
       that key in the loaded graph.  On mismatch: unlink the file and raise.
    6. ``shutil.rmtree(<adapter_dir>/<tier>)`` — same as before; the adapter
       slot is no longer the canonical store for this tier.

    Rollback: on step-5 sanity failure the graph.json is unlinked and
    ``RuntimeError`` is raised.  The adapter tier dir stays intact.

    Raises:
        _TierSkipped: When there are no active registry keys for this tier
            (clean state, nothing to migrate).
        paramem.graph.reconstruct.ReconstructionError: When any weight-probe
            fails under strict=True (the adapter's recall is below 1.0).
        RuntimeError: When the post-write sanity check fails (key present in
            registry but missing from the written graph).
    """
    from paramem.graph.reconstruct import ReconstructionError, reconstruct_graph
    from paramem.server.simulate_store import (
        _IK_KEY_ATTR,
        iter_quads,
        load_simulate_graph,
        save_simulate_graph,
    )

    # Step 1: check active keys for this tier.
    registry = loop.indexed_key_registry
    if registry is None:
        raise _TierSkipped(f"no indexed_key_registry on loop; tier {tier} skipped")
    active_keys = [k for k in registry.list_active() if registry.get_adapter_id(k) == tier]
    if not active_keys:
        raise _TierSkipped(f"no active registry keys for tier {tier}")

    # Step 2: reconstruct graph from weights (strict=True raises on any probe failure).
    try:
        result = reconstruct_graph(loop, tier=tier, strict=True)
    except ReconstructionError as exc:
        raise RuntimeError(
            f"train_to_simulate tier {tier}: weight reconstruction failed: {exc}"
        ) from exc
    graph = result.graph

    # Step 3: decorate edges with speaker_id + first_seen_cycle from indexed_key_cache.
    cache = getattr(loop, "indexed_key_cache", {})
    for subject, obj, eid, data in graph.edges(keys=True, data=True):
        ik_key = data.get(_IK_KEY_ATTR)
        if ik_key is None:
            continue
        cache_entry = cache.get(ik_key, {})
        data["speaker_id"] = cache_entry.get("speaker_id", "")
        data["first_seen_cycle"] = cache_entry.get("first_seen_cycle", 0)

    # Step 4: persist graph.
    target_dir = Path(config.simulate_dir) / tier
    target_dir.mkdir(parents=True, exist_ok=True)
    target_graph = target_dir / "graph.json"
    save_simulate_graph(graph, target_graph)

    # Step 5: sanity-check — every active registry key must be in the graph.
    loaded = load_simulate_graph(target_graph)
    graph_keys = {q["key"] for q in iter_quads(loaded)}
    missing = [k for k in active_keys if k not in graph_keys]
    if missing:
        target_graph.unlink(missing_ok=True)
        raise RuntimeError(
            f"train_to_simulate tier {tier}: sanity check failed — {len(missing)} key(s) "
            f"missing from written graph: {missing[:5]!r}{'...' if len(missing) > 5 else ''}; "
            f"rolled back simulate-store write"
        )

    # Step 6: remove adapter tier dir (kp file + all slot subdirs + .pending).
    # Source authoritative state has been weight-reconstructed and graph-persisted.
    tier_dir = Path(config.adapter_dir) / tier
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    logger.info(
        "active_store_migration: tier %s migrated to simulate via weight reconstruction;"
        " %d keys; deleted adapter dir %s",
        tier,
        len(active_keys),
        tier_dir,
    )


def _tier_adapter_config(loop: "ConsolidationLoop", tier: str):
    """Resolve the per-tier ``AdapterConfig`` from the loop.

    Raises ``_TierSkipped`` when the tier is configured-out
    (``procedural_config is None`` after operator disabled procedural).
    """
    cfg_map = {
        "episodic": loop.episodic_config,
        "semantic": loop.semantic_config,
        "procedural": loop.procedural_config,
    }
    tier_config = cfg_map.get(tier)
    if tier_config is None:
        raise _TierSkipped(f"tier {tier} not enabled (config is None)")
    return tier_config


def _migrate_tier_simulate_to_train(
    loop: "ConsolidationLoop", config: "ServerConfig", tier: str
) -> None:
    """Read simulate-store graph.json → train into <tier> adapter →
    probe at threshold=1.0 → on pass, persist slot + delete source graph.

    Caller must hold the GPU lock — training and the recall probe both
    drive the model forward and would race STT/TTS otherwise.

    The simulate-mode store is always quad-only (``graph.json``); the
    ``ConsolidationScheduleConfig`` validator enforces
    ``mode=simulate → indexed_format=quad``.  All format-conditional
    imports are therefore dropped in this path — only the quad helpers
    are used.

    Sequence:

    1. Load source graph from ``<simulate>/<tier>/graph.json``; extract
       quad dicts via ``iter_quads``.
    2. Hot-load into ``loop.indexed_key_cache`` + register keys into
       ``loop.indexed_key_registry`` with ``adapter_id=tier`` so the
       recall probe can find them.
    3. Reset the tier adapter to LoRA-zero
       (``delete_adapter`` + ``create_adapter`` from the tier's config),
       then ``switch_adapter`` so training writes into this tier.
    4. ``format_quadruple_training`` + ``_indexed_dataset`` to build the
       HF dataset; gradient checkpointing toggled around the format call
       to mirror the existing post_session_train pattern.
    5. ``train_adapter`` with the tier's adapter config and the loop's
       configured num_epochs.
    6. Recall probe via ``loop._run_recall_sanity_probe(tier, keyed_pairs)``
       at threshold 1.0 — stricter than the 0.95 in-process default.
    7. On pass: ``atomic_save_adapter`` writes the slot under
       ``<adapter_dir>/<tier>/<ts>/`` + writes the kp file at the
       canonical tier-level path
       ``<adapter_dir>/<tier>/keyed_pairs.json`` (matches
       ``_save_adapters._write_kp`` layout). Delete the source
       ``<simulate>/<tier>/graph.json``.
    8. On fail: reset the adapter back to LoRA-zero so failure does not
       leave half-trained weights resident, raise ``RuntimeError``.
    """
    # Simulate-mode store is always graph.json (quad-only; enforced by the
    # ConsolidationScheduleConfig validator: mode=simulate requires indexed_format=quad).
    from paramem.adapters.manifest import build_manifest_for
    from paramem.models.loader import (
        atomic_save_adapter,
        create_adapter,
        switch_adapter,
    )
    from paramem.server.simulate_store import iter_quads, load_simulate_graph
    from paramem.training.keyed_pairs_io import write_keyed_pairs_quad as _wkp
    from paramem.training.quadruple_memory import (
        build_registry as _build_reg,
    )
    from paramem.training.quadruple_memory import (
        format_quadruple_training as _format_training,
    )
    from paramem.training.trainer import TrainingHooks
    from paramem.training.trainer import train_adapter as _train_adapter

    _indexed_format = "quad"  # simulate mode is always quad

    source_graph = Path(config.simulate_dir) / tier / "graph.json"
    if not source_graph.exists():
        raise _TierSkipped(f"no graph.json at {source_graph}")

    graph = load_simulate_graph(source_graph)
    keyed_pairs = list(iter_quads(graph))
    if not keyed_pairs:
        raise _TierSkipped(f"empty graph.json at {source_graph}")

    tier_config = _tier_adapter_config(loop, tier)

    # Step 2: hot-load into loop in-memory state so the recall probe (which
    # reads from loop.indexed_key_cache via build_registry) can find the keys.
    # Use loop._cache_entry for uniform shape (Option-B invariant) so every
    # downstream reader (promotion-match, sp_index, triple-lookup) can access
    # source_* aliases unconditionally in both modes.  Mirrors seed_episodic_cache
    # / seed_semantic_cache / seed_procedural_cache in consolidation.py.
    setattr(loop, f"{tier}_simhash", _build_reg(keyed_pairs))
    for kp in keyed_pairs:
        key = kp["key"]
        kp_subject = kp.get("subject") or kp.get("source_subject") or ""
        kp_predicate = kp.get("predicate") or kp.get("source_predicate") or ""
        kp_object = kp.get("object") or kp.get("source_object") or ""
        loop.indexed_key_cache[key] = loop._cache_entry(
            key=key,
            subject=kp_subject,
            predicate=kp_predicate,
            object=kp_object,
            speaker_id=kp.get("speaker_id", ""),
            first_seen_cycle=kp.get("first_seen_cycle", 0),
            question=kp.get("question"),
            answer=kp.get("answer"),
        )
        if loop.indexed_key_registry is not None and key not in loop.indexed_key_registry:
            loop.indexed_key_registry.add(key, adapter_id=tier)

    # Step 3: reset adapter to LoRA-zero so training starts from a clean state.
    if tier in loop.model.peft_config:
        loop.model.delete_adapter(tier)
    loop.model = create_adapter(loop.model, tier_config, tier)
    switch_adapter(loop.model, tier)

    # Step 4: build training dataset. Disable gradient checkpointing for the
    # tokenizer call (matches the post_session_train pattern at
    # consolidation.py:2692-2696); re-enable before training.
    loop._disable_gradient_checkpointing()
    examples = _format_training(keyed_pairs, loop.tokenizer, max_length=1024)
    if not examples:
        raise _TierSkipped(f"_format_training produced no examples for tier {tier}")
    dataset = loop._indexed_dataset(examples)
    loop._enable_gradient_checkpointing()
    training_config = loop._make_training_config(num_epochs=loop.training_config.num_epochs)

    # Step 5: train. Output dir under a migration-scoped subdir so checkpoint
    # debris doesn't pollute the main slot layout.
    recall_cb = loop._maybe_make_recall_callback(
        keyed_pairs=keyed_pairs,
        adapter_name=tier,
        output_dir=Path(config.adapter_dir) / "active_store_migration" / tier,
        phase_name=f"migrate-{tier}",
    )
    _train_adapter(
        model=loop.model,
        tokenizer=loop.tokenizer,
        train_dataset=dataset,
        adapter_name=tier,
        training_config=training_config,
        adapter_config=tier_config,
        wandb_config=loop.wandb_config,
        output_dir=Path(config.adapter_dir) / "active_store_migration" / tier,
        run_name=f"migrate-simulate-to-train-{tier}",
        thermal_policy=loop._thermal_policy,
        hooks=TrainingHooks(on_shutdown_check=lambda: loop.shutdown_requested),
        callbacks_extra=[recall_cb] if recall_cb is not None else None,
    )

    # Step 6: recall probe at threshold=1.0 (stricter than 0.95 default).
    # _run_recall_sanity_probe already branches on loop._is_quad so the probe
    # harness matches the format the adapter was trained in.
    recall = loop._run_recall_sanity_probe(tier, keyed_pairs)
    if recall < 1.0:
        # Rollback: reset adapter to LoRA-zero. The slot was not yet saved to
        # disk so there's nothing to delete on the filesystem side.
        loop.model.delete_adapter(tier)
        loop.model = create_adapter(loop.model, tier_config, tier)
        raise RuntimeError(
            f"simulate_to_train tier {tier} recall {recall:.3f} < 1.0; "
            f"rolled back trained adapter to LoRA-zero"
        )

    # Step 7a: atomic-save the slot. Manifest building can fail (e.g. base-model
    # hash unavailable); we save without manifest in that case so the weights
    # are durable even when the manifest sidecar isn't.
    fingerprint_cache = getattr(loop, "fingerprint_cache", None)
    try:
        manifest = build_manifest_for(
            loop.model,
            loop.tokenizer,
            tier,
            registry_path=None,
            keyed_pairs_path=Path(config.adapter_dir) / tier / "keyed_pairs.json",
            key_count=len(keyed_pairs),
            base_model_hash_cache=fingerprint_cache,
            adapter_root=Path(config.adapter_dir),
            indexed_format=_indexed_format,
        )
    except Exception:
        logger.warning(
            "active_store_migration: tier %s manifest build failed — saving without manifest",
            tier,
        )
        manifest = None
    slot_path = atomic_save_adapter(
        loop.model,
        Path(config.adapter_dir) / tier,
        tier,
        manifest=manifest,
    )

    # Step 7b: write the canonical tier-level keyed_pairs.json.
    kp_path = Path(config.adapter_dir) / tier / "keyed_pairs.json"
    _wkp(kp_path, keyed_pairs)

    # Step 7c: persist the per-tier SimHash registry next to the adapter.
    # Mirrors training/consolidation.py::_save_adapters — without this file the
    # boot-time _load_simhash_registry returns an empty {} for this tier, and
    # probe_quad / probe_key then treat every recalled key as untrained
    # (verify_confidence → 0.0 → low_confidence:0.000), so every personal query
    # silently abstains even though the adapter recalls correctly. The simhash
    # was already built in Step 2 (setattr loop.<tier>_simhash); persist it as
    # {key: int} (same shape build_registry / build_registry_quad emit and that
    # _load_simhash_registry / get_simhash expect).
    from paramem.training.indexed_memory import save_registry as _save_registry

    _simhash = getattr(loop, f"{tier}_simhash", None)
    if _simhash:
        _save_registry(
            _simhash,
            Path(config.adapter_dir) / f"simhash_registry_{tier}.json",
        )

    # Step 7d: delete source (target is now authoritative + probe-confirmed).
    source_graph.unlink()

    logger.info(
        "active_store_migration: tier %s migrated to train; slot=%s, %d keys",
        tier,
        slot_path,
        len(keyed_pairs),
    )
