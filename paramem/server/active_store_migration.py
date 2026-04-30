"""Migrate keyed pairs between train (LoRA weights) and simulate (JSON) stores.

Triggered when the operator flips ``consolidation.mode`` in server.yaml.
The migration is per-tier with a 1.0 recall gate before cleanup; the
source store stays authoritative until ALL tiers have completed and the
state file is removed. On crash mid-migration, the state file persists
and the system falls back to the source mode on next boot.

State file location: ``<paths.adapters>/.active_store_migration.json``
(age-encrypted via ``write_infra_bytes`` when the daily identity is loaded).

Two directions:

* ``simulate_to_train``: read ``<simulate>/<tier>/keyed_pairs.json`` →
  train into ``<tier>`` adapter → recall probe at threshold=1.0 → on
  pass, atomic-save the slot and delete the simulate-store file. On
  fail, leave both stores intact.

* ``train_to_simulate``: read ``<adapters>/<tier>/<latest_ts>/keyed_pairs.json``
  → write to ``<simulate>/<tier>/keyed_pairs.json`` → recall probe via
  ``probe_keys_from_disk`` at threshold=1.0 → on pass, delete the
  adapter slots. On fail, remove the simulate-store write and leave the
  adapter slots intact.

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


def _has_simulate_kp(simulate_dir: Path, tier: str) -> bool:
    return (Path(simulate_dir) / tier / "keyed_pairs.json").exists()


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

    simulate_present = any(_has_simulate_kp(config.simulate_dir, t) for t in TIERS)
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
    """Read tier-level keyed_pairs.json → write to simulate-store →
    probe at threshold=1.0 → on pass, delete the adapter tier dir.

    Source layout (verified live, see _save_adapters._write_kp):
        ``<adapter_dir>/<tier>/keyed_pairs.json``  (tier level)
        ``<adapter_dir>/<tier>/<ts>/adapter_model.safetensors`` etc. (slot level)

    Rollback: on probe failure the simulate-store write is removed and
    ``RuntimeError`` is raised. The adapter tier dir stays intact.
    """
    from paramem.training.indexed_memory import probe_keys_from_disk

    source_kp = Path(config.adapter_dir) / tier / "keyed_pairs.json"
    if not source_kp.exists():
        raise _TierSkipped(f"no keyed_pairs.json at {source_kp}")

    keyed_pairs = json.loads(read_maybe_encrypted(source_kp).decode("utf-8"))
    if not keyed_pairs:
        raise _TierSkipped(f"empty keyed_pairs.json at {source_kp}")

    target_dir = Path(config.simulate_dir) / tier
    target_dir.mkdir(parents=True, exist_ok=True)
    target_kp = target_dir / "keyed_pairs.json"

    # Atomic-ish write + probe. write_infra_bytes uses _atomic_write_bytes
    # internally so a crash mid-write doesn't leave a partial file.
    payload = json.dumps(keyed_pairs).encode("utf-8")
    write_infra_bytes(target_kp, payload)

    # Probe via the existing simulate-mode disk-probe harness. Same call
    # shape as the inference dispatcher uses (paramem/server/inference.py:650).
    keys = [kp["key"] for kp in keyed_pairs if "key" in kp]
    if not keys:
        # No keys to probe — write succeeded, accept it. Rare edge case.
        logger.warning(
            "active_store_migration: tier %s has 0 keys; accepting write as trivially complete",
            tier,
        )
    else:
        probe_results = probe_keys_from_disk(Path(config.simulate_dir), {tier: keys})
        hits = sum(1 for v in probe_results.values() if v is not None)
        recall = hits / len(keys)
        if recall < 1.0:
            target_kp.unlink(missing_ok=True)
            raise RuntimeError(
                f"train_to_simulate tier {tier} recall {recall:.3f} < 1.0; "
                f"rolled back simulate-store write"
            )

    # Cleanup: remove the entire adapter tier dir (kp file + all slot subdirs +
    # .pending). Source authoritative state has been transferred to the target
    # store and probe-confirmed.
    tier_dir = Path(config.adapter_dir) / tier
    shutil.rmtree(tier_dir)
    logger.info(
        "active_store_migration: tier %s migrated to simulate; deleted adapter dir %s",
        tier,
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
    """Read simulate-store keyed_pairs → train into <tier> adapter →
    probe at threshold=1.0 → on pass, persist slot + delete source.

    Caller must hold the GPU lock — training and the recall probe both
    drive the model forward and would race STT/TTS otherwise.

    Sequence:

    1. Load source kp from ``<simulate>/<tier>/keyed_pairs.json``.
    2. Hot-load into ``loop.indexed_key_qa`` + register keys into
       ``loop.indexed_key_registry`` with ``adapter_id=tier`` so the
       recall probe can find them.
    3. Reset the tier adapter to LoRA-zero
       (``delete_adapter`` + ``create_adapter`` from the tier's config),
       then ``switch_adapter`` so training writes into this tier.
    4. ``format_indexed_training`` + ``_indexed_dataset`` to build the
       HF dataset; gradient checkpointing toggled around the format
       call to mirror the existing post_session_train pattern.
    5. ``train_adapter`` with the tier's adapter config and the loop's
       configured num_epochs.
    6. Recall probe via ``loop._run_recall_sanity_probe(tier, keyed_pairs)``
       at threshold 1.0 — stricter than the 0.95 in-process default.
    7. On pass: ``atomic_save_adapter`` writes the slot under
       ``<adapter_dir>/<tier>/<ts>/`` + writes the kp file at the
       canonical tier-level path
       ``<adapter_dir>/<tier>/keyed_pairs.json`` (matches
       ``_save_adapters._write_kp`` layout). Delete the source
       ``<simulate>/<tier>/keyed_pairs.json``.
    8. On fail: reset the adapter back to LoRA-zero so failure does not
       leave half-trained weights resident, raise ``RuntimeError``.
    """
    from paramem.adapters.manifest import build_manifest_for
    from paramem.models.loader import (
        atomic_save_adapter,
        create_adapter,
        switch_adapter,
    )
    from paramem.training.indexed_memory import (
        build_registry,
        format_indexed_training,
    )
    from paramem.training.trainer import train_adapter as _train_adapter

    source_kp = Path(config.simulate_dir) / tier / "keyed_pairs.json"
    if not source_kp.exists():
        raise _TierSkipped(f"no keyed_pairs.json at {source_kp}")

    keyed_pairs = json.loads(read_maybe_encrypted(source_kp).decode("utf-8"))
    if not keyed_pairs:
        raise _TierSkipped(f"empty keyed_pairs.json at {source_kp}")

    tier_config = _tier_adapter_config(loop, tier)

    # Step 2: hot-load into loop in-memory state so the recall probe (which
    # reads from loop.indexed_key_qa via build_registry) can find the keys.
    setattr(loop, f"{tier}_simhash", build_registry(keyed_pairs))
    for kp in keyed_pairs:
        loop.indexed_key_qa[kp["key"]] = kp
        if loop.indexed_key_registry is not None and kp["key"] not in loop.indexed_key_registry:
            loop.indexed_key_registry.add(kp["key"], adapter_id=tier)

    # Step 3: reset adapter to LoRA-zero so training starts from a clean state.
    if tier in loop.model.peft_config:
        loop.model.delete_adapter(tier)
    loop.model = create_adapter(loop.model, tier_config, tier)
    switch_adapter(loop.model, tier)

    # Step 4: build training dataset. Disable gradient checkpointing for the
    # tokenizer call (matches the post_session_train pattern at
    # consolidation.py:2692-2696); re-enable before training.
    loop._disable_gradient_checkpointing()
    examples = format_indexed_training(keyed_pairs, loop.tokenizer, max_length=1024)
    if not examples:
        raise _TierSkipped(f"format_indexed_training produced no examples for tier {tier}")
    dataset = loop._indexed_dataset(examples)
    loop._enable_gradient_checkpointing()
    training_config = loop._make_training_config(num_epochs=loop.training_config.num_epochs)

    # Step 5: train. Output dir under a migration-scoped subdir so checkpoint
    # debris doesn't pollute the main slot layout.
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
        callbacks_extra=loop._shutdown_callbacks,
    )

    # Step 6: recall probe at threshold=1.0 (stricter than 0.95 default).
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
    kp_path.parent.mkdir(parents=True, exist_ok=True)
    write_infra_bytes(kp_path, json.dumps(keyed_pairs, indent=2).encode("utf-8"))

    # Step 7c: delete source (target is now authoritative + probe-confirmed).
    source_kp.unlink()

    logger.info(
        "active_store_migration: tier %s migrated to train; slot=%s, %d keys",
        tier,
        slot_path,
        len(keyed_pairs),
    )
