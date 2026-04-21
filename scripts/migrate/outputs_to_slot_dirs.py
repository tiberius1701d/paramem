"""Migration script: reshape old flat adapter directories to slot-dir layout.

Old layout (pre-Slice-3a)::

    outputs/test8/mistral/20260101-120000/episodic/
        adapter_config.json
        adapter_model.safetensors
        keyed_pairs.json           # optional

New layout (post-Slice-3a)::

    outputs/test8/mistral/20260101-120000/episodic/
        20260101-120000/           # timestamped slot (mtime of safetensors)
            adapter_config.json
            adapter_model.safetensors
            keyed_pairs.json
            meta.json              # synthesized manifest

Discovery: a directory is old-layout iff it contains both
``adapter_config.json`` and ``adapter_model.safetensors`` at the same level.

Idempotency: already-migrated directories are skipped when:
- The parent directory name matches ``YYYYMMDD-HHMMSS`` (already a slot), or
- ``meta.json`` is already present at the top level.

Concurrency guard: if any training process is alive (detected via ``pgrep``)
and ``--force`` is not passed, the script exits with a helpful message.
``--dry-run`` skips the liveness check entirely (read-only, safe to run
alongside training for inspection).

Usage::

    python scripts/migrate/outputs_to_slot_dirs.py \\
        --outputs-root outputs/ \\
        [--registry-path path/to/indexed_key_registry.json] \\
        [--name-from-config] \\
        [--dry-run] \\
        [--verbose] \\
        [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import timezone
from pathlib import Path
from typing import Optional

# Allow running directly from the repo root without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paramem.adapters.manifest import (
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    MANIFEST_SCHEMA_VERSION,
    TokenizerFingerprint,
    write_manifest,
)

logger = logging.getLogger("migrate_outputs")

# Regex for YYYYMMDD-HHMMSS slot directory names.
_SLOT_TS_RE = re.compile(r"^\d{8}-\d{6}$")

# pgrep patterns covering training scripts (same set as TEST_PGREP in
# training-control.sh).  Add new test scripts here when registered there.
_TRAINING_PGREP_PATTERNS: list[str] = [
    "test4b_",
    "test6_",
    "test8_",
    "test10_",
    "test11_",
    "test13_",
]


def _pgrep_alive(patterns: list[str]) -> list[tuple[str, str]]:
    """Return (pid, pattern) pairs for any alive training process.

    Args:
        patterns: List of pgrep pattern strings.

    Returns:
        List of (pid_str, pattern) tuples for running matches.
    """
    alive = []
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for pid in result.stdout.strip().splitlines():
                    if pid.strip():
                        alive.append((pid.strip(), pattern))
        except (OSError, subprocess.TimeoutExpired):
            pass
    return alive


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file's bytes.

    Args:
        path: Path to file.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_sibling_registry(adapter_dir: Path, registry_path_override: Optional[Path]) -> str:
    """Locate a registry file and return its SHA-256 hex, or UNKNOWN.

    Looks in the following order:
    1. ``registry_path_override`` (CLI flag).
    2. ``<adapter_dir.parent>/indexed_key_registry.json``
    3. ``<adapter_dir.parent.parent>/indexed_key_registry.json``
    4. Any ``simhash_registry_*.json`` in the same locations.

    Args:
        adapter_dir: The old flat adapter directory.
        registry_path_override: Optional explicit path from CLI.

    Returns:
        SHA-256 hex of the registry file, or ``UNKNOWN``.
    """
    if registry_path_override is not None and registry_path_override.exists():
        return _sha256_file(registry_path_override)

    candidates = [
        adapter_dir.parent / "indexed_key_registry.json",
        adapter_dir.parent.parent / "indexed_key_registry.json",
    ]
    # Also try simhash registry as fallback.
    for parent in (adapter_dir.parent, adapter_dir.parent.parent):
        for p in parent.glob("simhash_registry_*.json"):
            candidates.append(p)

    for c in candidates:
        if c.exists():
            return _sha256_file(c)
    return UNKNOWN


def _find_sibling_keyed_pairs(adapter_dir: Path) -> tuple[str, int | str]:
    """Locate keyed_pairs.json inside the adapter dir and return (sha256, count).

    Args:
        adapter_dir: The old flat adapter directory.

    Returns:
        Tuple of (sha256_hex_or_UNKNOWN, key_count_or_UNKNOWN).
    """
    kp_path = adapter_dir / "keyed_pairs.json"
    if not kp_path.exists():
        return UNKNOWN, UNKNOWN
    try:
        data = json.loads(kp_path.read_bytes())
        count = len(data) if isinstance(data, list) else UNKNOWN
    except (json.JSONDecodeError, OSError):
        count = UNKNOWN
    return _sha256_file(kp_path), count


def _adapter_name_from_dir(adapter_dir: Path, *, name_from_config: bool) -> str:
    """Derive the adapter name from the directory or its config.

    Heuristic:
    - If ``adapter_dir.name == "adapter"``, return ``adapter_dir.parent.name``
      (handles nested paths like ``episodic/adapter/``).
    - Otherwise return ``adapter_dir.name``.
    - With ``--name-from-config``, read ``base_model_name_or_path`` from
      ``adapter_config.json`` and use it as the name if available.

    Args:
        adapter_dir: The old flat adapter directory.
        name_from_config: If ``True``, try reading the name from
            ``adapter_config.json``.

    Returns:
        Adapter name string.
    """
    if name_from_config:
        cfg_path = adapter_dir / "adapter_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                # PEFT adapter_config.json has no "name" field; use
                # base_model_name_or_path as a fallback repo identifier.
                name = cfg.get("adapter_name") or cfg.get("name")
                if name:
                    return name
            except (json.JSONDecodeError, OSError):
                pass

    if adapter_dir.name == "adapter":
        return adapter_dir.parent.name
    return adapter_dir.name


def _synthesize_manifest(
    adapter_dir: Path,
    *,
    registry_path_override: Optional[Path],
    name_from_config: bool,
) -> AdapterManifest:
    """Build a synthesized manifest for an old-layout adapter directory.

    All fingerprint fields that cannot be computed without reloading model
    weights are set to ``UNKNOWN``.  ``synthesized=True`` is always set so
    the startup validator shows yellow severity (not red) for these fields.

    Args:
        adapter_dir: The old flat adapter directory.
        registry_path_override: Optional explicit registry path from CLI.
        name_from_config: Derive adapter name from adapter_config.json.

    Returns:
        A fully-populated :class:`~paramem.adapters.manifest.AdapterManifest`
        with ``synthesized=True``.
    """
    from datetime import datetime

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    trained_at = (
        datetime.fromtimestamp(safetensors_path.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        if safetensors_path.exists()
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # --- base model ---
    base_repo = UNKNOWN
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            base_repo = cfg.get("base_model_name_or_path") or UNKNOWN
        except (json.JSONDecodeError, OSError):
            pass

    base_model_fp = BaseModelFingerprint(repo=base_repo, sha=UNKNOWN, hash=UNKNOWN)

    # --- tokenizer ---
    tokenizer_fp = TokenizerFingerprint(
        name_or_path=UNKNOWN,
        vocab_size=UNKNOWN,
        merges_hash=UNKNOWN,
    )

    # --- LoRA shape ---
    lora_rank = 0
    lora_alpha = 0
    lora_dropout = 0.0
    lora_targets: tuple[str, ...] = ()
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            lora_rank = int(cfg.get("r", 0))
            lora_alpha = int(cfg.get("lora_alpha", 0))
            lora_dropout = float(cfg.get("lora_dropout", 0.0))
            targets = cfg.get("target_modules", [])
            lora_targets = tuple(sorted(targets)) if targets else ()
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    lora_shape = LoRAShape(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_targets,
    )

    # --- registry + keyed_pairs ---
    registry_sha256 = _find_sibling_registry(adapter_dir, registry_path_override)
    keyed_pairs_sha256, key_count = _find_sibling_keyed_pairs(adapter_dir)

    name = _adapter_name_from_dir(adapter_dir, name_from_config=name_from_config)

    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=name,
        trained_at=trained_at,
        base_model=base_model_fp,
        tokenizer=tokenizer_fp,
        lora=lora_shape,
        registry_sha256=registry_sha256,
        keyed_pairs_sha256=keyed_pairs_sha256,
        key_count=key_count,
        synthesized=True,
    )


def _is_old_layout(candidate: Path) -> bool:
    """Return True if *candidate* is an old-layout flat adapter directory.

    A directory is old-layout iff:
    - It contains both ``adapter_config.json`` and
      ``adapter_model.safetensors`` at the same level.
    - Its name does NOT match ``YYYYMMDD-HHMMSS`` (already a slot).
    - ``meta.json`` is NOT present (already migrated).

    Args:
        candidate: Directory to inspect.

    Returns:
        ``True`` if the directory should be migrated.
    """
    if not candidate.is_dir():
        return False
    if _SLOT_TS_RE.match(candidate.name):
        return False
    if (candidate / "meta.json").exists():
        return False
    return (candidate / "adapter_config.json").exists() and (
        candidate / "adapter_model.safetensors"
    ).exists()


def _discover_old_layout_dirs(outputs_root: Path) -> list[Path]:
    """Walk *outputs_root* and return all old-layout adapter directories.

    Args:
        outputs_root: Root directory to search (typically ``outputs/``).

    Returns:
        Sorted list of old-layout directories found.
    """
    results = []
    for root, dirs, files in os.walk(outputs_root):
        # Skip hidden directories and slot dirs to avoid descending into them.
        dirs[:] = [d for d in dirs if not d.startswith(".") and not _SLOT_TS_RE.match(d)]
        p = Path(root)
        if _is_old_layout(p):
            results.append(p)
    return sorted(results)


def _reshape_dir(
    adapter_dir: Path,
    *,
    registry_path_override: Optional[Path],
    name_from_config: bool,
    dry_run: bool,
    verbose: bool,
) -> bool:
    """Reshape one old-layout directory to slot layout.

    1. Derive slot timestamp from ``adapter_model.safetensors`` mtime.
    2. Synthesize manifest.
    3. ``os.rename(adapter_dir, slot_path)``.
    4. ``write_manifest(slot_path, manifest)``.

    Args:
        adapter_dir: Old-layout flat adapter directory.
        registry_path_override: Optional explicit registry path.
        name_from_config: Derive name from adapter_config.json.
        dry_run: Log action but skip filesystem mutations.
        verbose: Log extra detail.

    Returns:
        ``True`` if the directory was (or would be) migrated, ``False`` if
        skipped.
    """
    from datetime import datetime

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    mtime = safetensors_path.stat().st_mtime
    slot_ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    slot_path = adapter_dir.parent / slot_ts

    if verbose:
        logger.info("Migrating %s → %s", adapter_dir, slot_path)

    if dry_run:
        logger.info("[dry-run] Would rename %s → %s", adapter_dir, slot_path)
        return True

    manifest = _synthesize_manifest(
        adapter_dir,
        registry_path_override=registry_path_override,
        name_from_config=name_from_config,
    )

    # Handle slot collision: append counter suffix.
    final_slot = slot_path
    counter = 1
    while final_slot.exists():
        final_slot = adapter_dir.parent / f"{slot_ts}-{counter}"
        counter += 1

    adapter_dir.rename(final_slot)
    write_manifest(final_slot, manifest)
    logger.info("Migrated %s → %s", adapter_dir, final_slot)
    return True


def migrate(
    outputs_root: Path,
    *,
    registry_path: Optional[Path] = None,
    name_from_config: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    force: bool = False,
) -> int:
    """Run the migration.

    Args:
        outputs_root: Root of the outputs directory tree.
        registry_path: Optional explicit registry path to use for all
            ``registry_sha256`` fields.
        name_from_config: Derive adapter name from ``adapter_config.json``.
        dry_run: Read-only mode; no filesystem mutations.  Skips liveness
            check.
        verbose: Log extra detail per directory.
        force: Proceed even if training processes are alive.

    Returns:
        Exit code (0 = success, 1 = blocked by alive processes).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not dry_run:
        alive = _pgrep_alive(_TRAINING_PGREP_PATTERNS)
        if alive and not force:
            pids = ", ".join(f"PID {pid} ({pat})" for pid, pat in alive)
            logger.error(
                "ERROR: Training in progress (%s). "
                "Pause first with `tpause`, or rerun with --force (not recommended).",
                pids,
            )
            return 1
        if alive and force:
            pids = ", ".join(f"PID {pid} ({pat})" for pid, pat in alive)
            logger.warning("--force: proceeding with alive training processes: %s", pids)

    if not outputs_root.exists():
        logger.error("outputs_root %s does not exist", outputs_root)
        return 1

    old_dirs = _discover_old_layout_dirs(outputs_root)
    if not old_dirs:
        logger.info("No old-layout adapter directories found under %s", outputs_root)
        return 0

    logger.info("Found %d old-layout director%s", len(old_dirs), "y" if len(old_dirs) == 1 else "ies")
    migrated = 0
    for adapter_dir in old_dirs:
        ok = _reshape_dir(
            adapter_dir,
            registry_path_override=registry_path,
            name_from_config=name_from_config,
            dry_run=dry_run,
            verbose=verbose,
        )
        if ok:
            migrated += 1

    logger.info(
        "%s %d director%s",
        "Would migrate" if dry_run else "Migrated",
        migrated,
        "y" if migrated == 1 else "ies",
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root of the outputs tree (default: outputs/).",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help=(
            "Explicit path to indexed_key_registry.json.  When given, every "
            "synthesized manifest uses this registry's SHA-256.  Otherwise the "
            "script looks for a sibling registry file."
        ),
    )
    parser.add_argument(
        "--name-from-config",
        action="store_true",
        help="Derive adapter name from adapter_config.json instead of directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes.  Skips liveness check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log extra detail.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if training processes are alive (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI invocation."""
    args = _parse_args()
    sys.exit(
        migrate(
            Path(args.outputs_root),
            registry_path=Path(args.registry_path) if args.registry_path else None,
            name_from_config=args.name_from_config,
            dry_run=args.dry_run,
            verbose=args.verbose,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
