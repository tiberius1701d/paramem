#!/usr/bin/env python3
"""One-shot migration: relocate donor checkpoints into adapter stores.

Donors used to live in a private tree keyed by topology alone

    <adapter_dir>/_donor/<topology_id>/<slot_date>/

which no backup, restore or retention routine recognised — the restore
sweep deleted them as strays, and two base models sharing a topology
overwrote one another.  They are now ordinary adapter stores, siblings of
the main tiers, keyed by (base model, topology)

    <adapter_dir>/donor-<topology_id>-b<base8>/<slot_date>/

each carrying the same ``meta.json`` manifest every other store carries.

This script writes that manifest and moves each slot.  Nothing is
retrained: the manifest's base-model and tokenizer fingerprints are copied
from an existing main-tier manifest under the same adapter root that
records the SAME ``base_model.repo`` as the donor — the fingerprint is a
property of the model, so this is the value the donor's own build would
have computed, not an approximation.  A donor with no matching tier
manifest is left in place and reported: relocating it would mean inventing
a fingerprint.

Idempotent: a clean re-run is a no-op.

Usage:
    python scripts/migrate/relocate_donor_adapters.py [--adapter-dir PATH] [--dry-run]

Defaults to ``data/ha/adapters`` (the live host's directory).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    TokenizerFingerprint,
    is_slot_name,
    write_manifest,
)
from paramem.training.donor import DONOR_META_FILENAME, donor_store_dir

logger = logging.getLogger(__name__)

LEGACY_DONOR_DIRNAME = "_donor"


def _iter_legacy_slots(adapter_dir: Path) -> list[Path]:
    """Return every ``_donor/<topology_id>/<slot_date>/`` directory."""
    legacy_root = adapter_dir / LEGACY_DONOR_DIRNAME
    if not legacy_root.is_dir():
        return []
    slots: list[Path] = []
    for topology_dir in sorted(legacy_root.iterdir()):
        if not topology_dir.is_dir() or topology_dir.name.startswith("."):
            continue
        for slot in sorted(topology_dir.iterdir()):
            if slot.is_dir() and is_slot_name(slot.name):
                slots.append(slot)
    return slots


def _tier_fingerprints(adapter_dir: Path, base_repo: str) -> tuple[dict, dict] | None:
    """Return ``(base_model, tokenizer)`` dicts from a tier manifest for *base_repo*.

    Scans every main-tier slot manifest under *adapter_dir* for one whose
    ``base_model.repo`` equals *base_repo* and whose fingerprint fields are
    fully populated.  Returns ``None`` when none matches — the caller then
    leaves that donor alone rather than inventing a fingerprint.
    """
    for tier in ("episodic", "semantic", "procedural"):
        tier_dir = adapter_dir / tier
        if not tier_dir.is_dir():
            continue
        for slot in sorted(tier_dir.iterdir(), reverse=True):
            meta_path = slot / "meta.json"
            if not slot.is_dir() or not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                continue
            base_model = meta.get("base_model") or {}
            tokenizer = meta.get("tokenizer") or {}
            if base_model.get("repo") != base_repo:
                continue
            if not base_model.get("sha") or not base_model.get("hash"):
                continue
            return base_model, tokenizer
    return None


def _manifest_for(
    slot: Path, donor_meta: dict, base_model: dict, tokenizer: dict
) -> AdapterManifest:
    """Build the donor slot's manifest from on-disk facts only.

    ``lora`` comes from the slot's own ``adapter_config.json``; ``key_count``
    from the donor's recorded triple set; ``trained_at`` from the slot stamp.
    ``registry_sha256`` is empty because a donor carries no key registry —
    the value ``find_live_slot`` matches a donor slot on.
    """
    adapter_config = json.loads((slot / "adapter_config.json").read_text())
    stamp = slot.name  # YYYYMMDD-HHMMSS
    trained_at = (
        f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
    )
    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=slot.parent.name,
        trained_at=trained_at,
        base_model=BaseModelFingerprint(
            repo=base_model["repo"], sha=base_model["sha"], hash=base_model["hash"]
        ),
        tokenizer=TokenizerFingerprint(
            name_or_path=tokenizer.get("name_or_path", base_model["repo"]),
            vocab_size=tokenizer.get("vocab_size", "UNKNOWN"),
            merges_hash=tokenizer.get("merges_hash", "UNKNOWN"),
        ),
        lora=LoRAShape(
            rank=int(adapter_config["r"]),
            alpha=int(adapter_config["lora_alpha"]),
            dropout=float(adapter_config.get("lora_dropout", 0.0)),
            target_modules=tuple(sorted(adapter_config.get("target_modules") or [])),
        ),
        registry_sha256="",
        key_count=len(donor_meta.get("triples") or []),
        synthesized=False,
        window_stamp="",
    )


def relocate(adapter_dir: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Relocate every legacy donor slot.  Returns ``(moved, skipped)``."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")

    slots = _iter_legacy_slots(adapter_dir)
    if not slots:
        logger.info("No legacy donor slots found at %s — clean layout", adapter_dir)
        return 0, 0

    moved = 0
    skipped = 0
    for slot in slots:
        meta_path = slot / DONOR_META_FILENAME
        if not meta_path.exists():
            logger.warning("Skipping %s: no %s", slot, DONOR_META_FILENAME)
            skipped += 1
            continue
        donor_meta = json.loads(meta_path.read_text())
        base_repo = donor_meta.get("base_model_id")
        lora_shape = donor_meta.get("lora_shape")
        if not base_repo or not lora_shape:
            logger.warning("Skipping %s: donor meta records no base model / lora shape", slot)
            skipped += 1
            continue

        fingerprints = _tier_fingerprints(adapter_dir, base_repo)
        if fingerprints is None:
            logger.warning(
                "Skipping %s: no main-tier manifest under %s records base_model.repo=%r, "
                "so its fingerprint cannot be derived without loading the model",
                slot,
                adapter_dir,
                base_repo,
            )
            skipped += 1
            continue
        base_model, tokenizer = fingerprints

        store_dir = donor_store_dir(adapter_dir, base_repo, lora_shape)
        dst = store_dir / slot.name
        if dst.exists():
            logger.info("Skipping %s: destination %s already exists", slot, dst)
            skipped += 1
            continue

        manifest = _manifest_for(slot, donor_meta, base_model, tokenizer)
        # Drop the base model and LoRA shape from donor_meta: they now live in
        # the manifest, and a second recorded copy is a drift source.
        trimmed = {k: v for k, v in donor_meta.items() if k not in ("base_model_id", "lora_shape")}

        logger.info("%s → %s", slot, dst)
        if dry_run:
            moved += 1
            continue

        write_manifest(slot, manifest)
        meta_path.write_text(json.dumps(trimmed))
        store_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(slot), str(dst))
        moved += 1

    if not dry_run:
        _remove_legacy_tree_if_only_scaffolding(adapter_dir)

    return moved, skipped


def _remove_legacy_tree_if_only_scaffolding(adapter_dir: Path) -> None:
    """Drop ``_donor/`` once every checkpoint has left it.

    What a completed relocation leaves behind is dot-prefixed training
    scaffolding (``.training_scratch/`` holding ``progress.json`` /
    ``epoch_log.json``, and ``.pending/``) — regenerable, and the same
    category the backup path already refuses to capture.  Anything else
    still under the tree is an unrelocated checkpoint, so the tree stays and
    the operator is told what held it.
    """
    legacy_root = adapter_dir / LEGACY_DONOR_DIRNAME
    if not legacy_root.is_dir():
        return
    remaining = [
        child
        for topology_dir in legacy_root.iterdir()
        if topology_dir.is_dir()
        for child in topology_dir.iterdir()
        if not child.name.startswith(".")
    ]
    if remaining:
        logger.warning(
            "Leaving %s in place: %d entr%s not relocated (%s)",
            legacy_root,
            len(remaining),
            "y" if len(remaining) == 1 else "ies",
            ", ".join(str(p.relative_to(legacy_root)) for p in remaining),
        )
        return
    scaffolding = sorted(p for p in legacy_root.rglob("*") if p.is_file())
    shutil.rmtree(legacy_root)
    logger.info(
        "Removed legacy donor tree %s (%d scaffolding file%s: %s)",
        legacy_root,
        len(scaffolding),
        "" if len(scaffolding) == 1 else "s",
        ", ".join(str(p.relative_to(legacy_root)) for p in scaffolding) or "none",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("data/ha/adapters"),
        help="Adapter root holding the tier stores (default: data/ha/adapters)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would move without touching the filesystem",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    moved, skipped = relocate(args.adapter_dir, dry_run=args.dry_run)
    logger.info("Done: %d relocated, %d skipped", moved, skipped)
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
