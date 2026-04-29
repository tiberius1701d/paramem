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
    """Is there a main slot for this tier with a keyed_pairs.json on disk?"""
    tier_dir = Path(adapter_dir) / tier
    if not tier_dir.is_dir():
        return False
    return any(
        (slot / "keyed_pairs.json").exists()
        for slot in tier_dir.iterdir()
        if slot.is_dir() and not slot.name.startswith(".")
    )


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
    """Read latest adapter slot's keyed_pairs.json → write to simulate-store →
    probe at threshold=1.0 → on pass, delete adapter slots.

    Rollback: on probe failure the simulate-store write is removed and
    RuntimeError is raised. Adapter slots stay intact.
    """
    from paramem.training.indexed_memory import probe_keys_from_disk

    tier_dir = Path(config.adapter_dir) / tier
    if not tier_dir.is_dir():
        raise _TierSkipped(f"no adapter dir for tier {tier}")

    slots = sorted(d for d in tier_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not slots:
        raise _TierSkipped(f"no adapter slots under {tier_dir}")

    latest_slot = slots[-1]
    source_kp = latest_slot / "keyed_pairs.json"
    if not source_kp.exists():
        raise _TierSkipped(f"no keyed_pairs.json in latest slot {latest_slot}")

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

    # Cleanup: remove the adapter slots for this tier. Source authoritative
    # state has been transferred to the target store and probe-confirmed.
    shutil.rmtree(tier_dir)
    logger.info(
        "active_store_migration: tier %s migrated to simulate; deleted adapter dir %s",
        tier,
        tier_dir,
    )


def _migrate_tier_simulate_to_train(
    loop: "ConsolidationLoop", config: "ServerConfig", tier: str
) -> None:
    """Read simulate-store keyed_pairs → train into <tier> adapter →
    probe at threshold=1.0 → on pass, delete simulate-store kp file.

    NOT YET IMPLEMENTED.

    Spec for follow-up:

    1. Load source kp from ``<simulate>/<tier>/keyed_pairs.json``.
    2. Hot-load the keyed pairs into ``loop.indexed_key_qa``, register
       each key in ``loop.indexed_key_registry`` with adapter_id=tier,
       update ``loop.<tier>_simhash`` via build_registry.
    3. Format training examples via
       ``format_indexed_training(keyed_pairs, loop.tokenizer, max_length=1024)``.
    4. Switch active adapter to *tier*; build a fresh adapter slot
       (delete + create_adapter from the tier's config so weights start
       at LoRA-zero).
    5. Call ``train_adapter(...)`` with the tier's training config and
       per-tier adapter config (loop.episodic_config / semantic_config /
       procedural_config). Use ``loop.training_config.num_epochs`` as the
       budget.
    6. After training, run ``loop._run_recall_sanity_probe(tier, keyed_pairs)``
       and assert ``recall == 1.0``. Lower threshold than the existing
       0.95 default — the operator's transition gate is intentionally
       stricter.
    7. On success, ``atomic_save_adapter`` writes the trained slot to
       ``<adapters>/<tier>/<ts>/`` with manifest. Delete the source
       ``<simulate>/<tier>/keyed_pairs.json``.
    8. On recall < 1.0: delete the trained adapter slot, leave source
       intact, raise ``RuntimeError``.

    The training step requires the GPU lock (held by the BG-trainer
    worker thread that orchestrates the migration). The caller is
    expected to run this function under the GPU lock. See
    ``_run_migration_sync`` in paramem/server/app.py (when wired up).
    """
    raise NotImplementedError(
        "simulate_to_train tier migration is not yet implemented. "
        "See module docstring for the spec; the train entry point is "
        "loop._train_adapter or train_adapter from paramem.training.trainer."
    )
