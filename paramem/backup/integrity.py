"""Infrastructure integrity checker for ParaMem.

Verifies the on-disk state of every tier's registry, simhash, manifest,
graph, key_metadata, and common files (speaker_profiles, observed_languages,
state/backup.json).

Public API
----------
- :class:`FileCheck` — result of checking one file.
- :class:`IntegrityReport` — aggregated result for the whole store.
- :func:`verify_infrastructure_integrity` — run the full suite and return
  an :class:`IntegrityReport`.

Exception→status mapping
-------------------------
The boundary ``try/except`` blocks in this module wrap ONLY the loader call
and name specific exception types.  They record a status and propagate the
detail as a human-readable string.  They do NOT silently swallow errors —
the ``ok``/``failures`` fields on :class:`IntegrityReport` expose every
non-ok non-skipped result to the caller.  Unexpected exceptions (loader bugs,
``OSError``, ``AttributeError``, etc.) propagate so the caller sees them
rather than a silent ``parse_error`` fallback.

Encryption handling
-------------------
``daily_loadable=False`` (default): age-encrypted files whose daily key is
absent report ``undecryptable`` with the distinct "daily identity not loaded"
detail.  These are NOT counted as corruption failures.  The caller decides
(see boot wiring).

``daily_loadable=True``: a decrypt failure IS corruption and IS counted as
a failure.

Interim slot enumeration
------------------------
Interim dirs are scanned under EVERY tier root (not only episodic) via
``rglob("interim_*")`` so future-tier interim slots are covered.  Main tiers
use :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name` for
their slot root; interim tiers use their on-disk dir directly.

Required-vs-optional matrix
----------------------------
- ``train`` mode: registry required when tier has keys, simhash required when
  registry non-empty, manifest required for the live weight slot.  Graph is
  optional/skipped in train mode.
- ``simulate`` mode: graph required for committed tier dirs, manifest optional.
- key_metadata, speaker_profiles, observed_languages, state/backup.json:
  always optional (skipped when absent — fresh installs lack them).
- Empty/absent semantic, absent interim, and partial interim slots (dir present
  but registry absent) → skipped, NOT a failure.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyrage

logger = logging.getLogger(__name__)

# ---- Status vocabulary ----
_OK = "ok"
_SKIPPED = "skipped"
_MISSING = "missing"
_UNDECRYPTABLE = "undecryptable"
_PARSE_ERROR = "parse_error"
_SCHEMA_ERROR = "schema_error"
_INCONSISTENT = "inconsistent"

# Statuses that count as failures (used to build IntegrityReport.failures).
_FAILURE_STATUSES = {_MISSING, _UNDECRYPTABLE, _PARSE_ERROR, _SCHEMA_ERROR, _INCONSISTENT}

# Detail string used for the "no daily key" undecryptable case — the boot
# wiring uses it to distinguish no-key from corruption.
_DETAIL_NO_KEY = "daily identity not loaded"
# Detail string for a ciphertext that fails decryption (wrong key or corrupt).
_DETAIL_BAD_KEY = "ciphertext corrupt or wrong key"


def _no_key_detail(exc: Exception) -> str:
    """Return ``_DETAIL_NO_KEY`` when *exc* indicates a missing daily identity.

    Used to normalise the detail string from RuntimeError raised by
    ``read_maybe_encrypted`` when the daily age identity is not loaded.

    Args:
        exc: The caught exception.

    Returns:
        ``_DETAIL_NO_KEY`` when the exception text matches a missing-identity
        pattern; otherwise ``str(exc)``.
    """
    msg = str(exc).lower()
    if "age envelope" in msg or "daily identity" in msg:
        return _DETAIL_NO_KEY
    return str(exc)


# Main tier names in the canonical order.
_MAIN_TIERS = ("episodic", "semantic", "procedural")


@dataclass(frozen=True)
class FileCheck:
    """Result of checking one infrastructure file.

    Attributes:
        path: String path of the checked file (str(Path), JSON-serializable).
        category: Logical category — one of ``"registry"``, ``"simhash"``,
            ``"manifest"``, ``"graph"``, ``"key_metadata"``, or ``"common"``.
        tier: Tier name (e.g. ``"episodic"``) or ``"common"`` for cross-tier
            files.
        status: One of ``"ok"``, ``"skipped"``, ``"missing"``,
            ``"undecryptable"``, ``"parse_error"``, ``"schema_error"``, or
            ``"inconsistent"``.
        detail: Human-readable explanation; empty string when status is ``"ok"``.
    """

    path: str
    category: str
    tier: str
    status: str
    detail: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "path": self.path,
            "category": self.category,
            "tier": self.tier,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class IntegrityReport:
    """Aggregated result of a full infrastructure integrity check.

    Attributes:
        ok: ``True`` when no failures are present (skipped entries are not
            failures).
        checks: All :class:`FileCheck` results including ok and skipped.
        failures: Subset of *checks* whose ``status`` is not ``"ok"`` or
            ``"skipped"``.
    """

    ok: bool
    checks: list[FileCheck]
    failures: list[FileCheck]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict suitable for the ``GET /integrity`` endpoint."""
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "failures": [c.to_dict() for c in self.failures],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_registry(path: Path, tier: str) -> tuple[FileCheck, list[str] | None]:
    """Check an ``indexed_key_registry.json`` via :class:`KeyRegistry.load`.

    Returns a ``(FileCheck, active_keys)`` pair.  ``active_keys`` is the list
    of active key names when the registry loaded successfully, ``None``
    otherwise.  Callers use the returned keys for cross-consistency checks to
    avoid a second read of the same file.

    Unexpected exceptions (loader bugs, ``OSError``, ``AttributeError``, etc.)
    propagate so the caller sees them rather than a silent fallback status.

    Returns:
        ``(FileCheck, active_keys | None)`` where ``active_keys`` is the list
        of active key strings on success, or ``None`` on any non-ok status.
    """
    from paramem.training.key_registry import KeyRegistry

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, "registry", tier, _SKIPPED, ""), None

    try:
        reg = KeyRegistry.load(path)
        active_keys = reg.list_active()
        return FileCheck(path_str, "registry", tier, _OK, ""), active_keys
    except RuntimeError as exc:
        # RuntimeError from read_maybe_encrypted means no daily identity loaded.
        return FileCheck(path_str, "registry", tier, _UNDECRYPTABLE, _no_key_detail(exc)), None
    except pyrage.DecryptError:
        return FileCheck(path_str, "registry", tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY), None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "registry", tier, _PARSE_ERROR, str(exc)), None
    except KeyError as exc:
        return FileCheck(path_str, "registry", tier, _SCHEMA_ERROR, str(exc)), None


def _check_simhash(path: Path, tier: str) -> tuple[FileCheck, dict | None]:
    """Check a ``simhash_registry.json`` via :func:`load_registry`.

    Returns a ``(FileCheck, simhash_dict)`` pair.  ``simhash_dict`` is the
    loaded mapping when the file parsed successfully, ``None`` otherwise.
    Callers use the returned dict for cross-consistency checks to avoid a
    second read of the same file.

    Unexpected exceptions propagate so the caller sees them rather than a
    silent fallback status.

    Returns:
        ``(FileCheck, simhash_dict | None)`` where ``simhash_dict`` is the
        loaded registry dict on success, or ``None`` on any non-ok status.
    """
    from paramem.memory.persistence import load_registry

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, "simhash", tier, _SKIPPED, ""), None

    try:
        sh_dict = load_registry(path)
        return FileCheck(path_str, "simhash", tier, _OK, ""), sh_dict
    except RuntimeError as exc:
        return FileCheck(path_str, "simhash", tier, _UNDECRYPTABLE, _no_key_detail(exc)), None
    except pyrage.DecryptError:
        return FileCheck(path_str, "simhash", tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY), None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "simhash", tier, _PARSE_ERROR, str(exc)), None


def _check_manifest(slot_dir: Path, tier: str) -> FileCheck:
    """Check a ``meta.json`` via :func:`paramem.adapters.manifest.read_manifest`.

    The manifest is PLAINTEXT — no decrypt branch.  Raises
    ``ManifestNotFoundError``/``ManifestSchemaError`` on failure.

    Returns:
        :class:`FileCheck` with category ``"manifest"`` and the resolved status.
    """
    from paramem.adapters.manifest import (
        ManifestNotFoundError,
        ManifestSchemaError,
        read_manifest,
    )

    meta_path = slot_dir / "meta.json"
    path_str = str(meta_path)

    try:
        read_manifest(slot_dir)
        return FileCheck(path_str, "manifest", tier, _OK, "")
    except ManifestNotFoundError:
        return FileCheck(path_str, "manifest", tier, _MISSING, "meta.json not found")
    except ManifestSchemaError as exc:
        return FileCheck(path_str, "manifest", tier, _SCHEMA_ERROR, str(exc))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "manifest", tier, _PARSE_ERROR, str(exc))


def _check_graph(path: Path, tier: str) -> FileCheck:
    """Check a ``graph.json`` via :func:`paramem.memory.persistence.load_memory_from_disk`.

    Graph checks do not contribute a parsed payload to cross-consistency checks
    so this helper returns a plain :class:`FileCheck` (not a tuple).

    Unexpected exceptions propagate so the caller sees them rather than a
    silent fallback status.

    Returns:
        :class:`FileCheck` with category ``"graph"`` and the resolved status.
    """
    from paramem.memory.persistence import load_memory_from_disk

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, "graph", tier, _MISSING, "graph.json not found")

    try:
        load_memory_from_disk(path)
        return FileCheck(path_str, "graph", tier, _OK, "")
    except RuntimeError as exc:
        return FileCheck(path_str, "graph", tier, _UNDECRYPTABLE, _no_key_detail(exc))
    except pyrage.DecryptError:
        return FileCheck(path_str, "graph", tier, _UNDECRYPTABLE, _DETAIL_BAD_KEY)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, "graph", tier, _PARSE_ERROR, str(exc))


def _check_common_file(path: Path, category: str) -> tuple[FileCheck, dict | None]:
    """Check a common JSON file via :func:`read_maybe_encrypted` + ``json.loads``.

    Returns a ``(FileCheck, parsed_dict)`` pair.  ``parsed_dict`` is the
    deserialized JSON object when the file loaded successfully, ``None``
    otherwise.  Callers use the returned dict for cross-consistency checks
    (e.g. key_metadata orphan check) to avoid a second read of the same file.

    Unexpected exceptions propagate so the caller sees them rather than a
    silent fallback status.

    Args:
        path: Path to check.
        category: One of ``"key_metadata"`` or ``"common"``.

    Returns:
        ``(FileCheck, parsed_dict | None)`` where ``parsed_dict`` is the
        deserialized JSON on success, or ``None`` on any non-ok status.
        Missing files return ``(_SKIPPED, None)`` (optional files — fresh
        installs lack them).
    """
    from paramem.backup.encryption import read_maybe_encrypted

    path_str = str(path)
    if not path.exists():
        return FileCheck(path_str, category, "common", _SKIPPED, ""), None

    try:
        raw = read_maybe_encrypted(path)
        parsed = json.loads(raw.decode("utf-8"))
        return FileCheck(path_str, category, "common", _OK, ""), parsed
    except RuntimeError as exc:
        return FileCheck(path_str, category, "common", _UNDECRYPTABLE, _no_key_detail(exc)), None
    except pyrage.DecryptError:
        return FileCheck(path_str, category, "common", _UNDECRYPTABLE, _DETAIL_BAD_KEY), None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileCheck(path_str, category, "common", _PARSE_ERROR, str(exc)), None


def _find_live_slot_for_tier(adapter_dir: Path, tier_name: str) -> Path | None:
    """Find the newest slot dir under *adapter_dir/<tier_name>* that has a meta.json.

    Used to locate the live weight slot for manifest checking in train mode.
    Returns the newest slot by mtime, or ``None`` when none exists.

    Args:
        adapter_dir: Root adapter directory.
        tier_name: Tier name (e.g. ``"episodic"``).

    Returns:
        Path to the slot directory, or ``None``.
    """
    tier_root = adapter_dir / tier_name
    if not tier_root.is_dir():
        return None
    candidates: list[Path] = []
    for entry in tier_root.iterdir():
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            continue
        if (entry / "meta.json").exists():
            candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_no_key_check(check: FileCheck) -> bool:
    """Return ``True`` when *check* is an undecryptable-with-no-key result.

    Args:
        check: The file check to inspect.

    Returns:
        ``True`` when the status is ``"undecryptable"`` and the detail
        indicates the daily key was not loaded.
    """
    return check.status == _UNDECRYPTABLE and check.detail == _DETAIL_NO_KEY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# The canonical 3-file signature of a committed adapter slot.  A subdir under
# <adapter_dir>/<tier>/ is considered complete only when ALL three are present.
# Missing any one of them marks the slot as partial-trained scratch and the
# boot housekeeping pass deletes it (see :func:`cleanup_partial_slots`).
_REQUIRED_SLOT_FILES: tuple[str, ...] = (
    "meta.json",
    "adapter_config.json",
    "adapter_model.safetensors",
)


def cleanup_partial_slots(adapter_dir: Path) -> list[dict]:
    """Delete partial-trained adapter slot directories under each main tier.

    Walks ``<adapter_dir>/<tier>/`` for every tier in :data:`_MAIN_TIERS` and
    removes any subdirectory that is NOT a complete slot.  A "complete slot"
    has all three files in :data:`_REQUIRED_SLOT_FILES`; missing any one of
    them marks it as scratch from an interrupted training run and the
    directory is removed via ``shutil.rmtree``.

    Skipped (never touched):
    - Dotted entries (``.quarantine``, ``.tmp``).
    - The staging slot conventions are in-memory PEFT keys, not on disk —
      this function cannot affect them.
    - The ``bg_checkpoint_epoch`` and ``checkpoint-*`` scratch dirs written
      by HF Trainer live UNDER the caller's ``output_dir`` (training-side),
      not under ``adapter_dir/<tier>/`` — they are out of scope.

    Args:
        adapter_dir: Root adapter directory (``config.adapter_dir`` /
            ``paths.adapters``).

    Returns:
        One dict per deleted slot describing what was removed, suitable for
        embedding in ``_state["integrity_cleanup"]`` and rendering by the
        attention populator::

            {"tier": "episodic",
             "slot_name": "interim_20260526T1200",
             "path": "/.../adapters/episodic/interim_20260526T1200",
             "missing": ["meta.json"]}

        Returns an empty list when no partial slots are found.
    """
    removed: list[dict] = []
    for tier_name in _MAIN_TIERS:
        tier_root = adapter_dir / tier_name
        if not tier_root.is_dir():
            continue
        for entry in tier_root.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            missing = [f for f in _REQUIRED_SLOT_FILES if not (entry / f).exists()]
            if not missing:
                continue  # complete slot — retained
            logger.warning(
                "integrity-cleanup: removing partial slot %s (missing: %s)",
                entry,
                ", ".join(missing),
            )
            shutil.rmtree(entry, ignore_errors=False)
            removed.append(
                {
                    "tier": tier_name,
                    "slot_name": entry.name,
                    "path": str(entry),
                    "missing": missing,
                }
            )
    return removed


def verify_infrastructure_integrity(
    config,
    *,
    store=None,
    daily_loadable: bool = False,
) -> IntegrityReport:
    """Run the full infrastructure integrity check and return an :class:`IntegrityReport`.

    Checks every tier's ``indexed_key_registry.json``, ``simhash_registry.json``,
    ``meta.json`` (live weight slot), and ``graph.json`` (simulate mode only).
    Also checks ``registry/key_metadata.json``, ``speaker_profiles.json``,
    ``observed_languages.json``, and ``state/backup.json``.

    Runs cross-consistency checks on tiers whose registry loaded ``"ok"``:
    registry keys vs simhash keys, and key_metadata orphans.

    Args:
        config: Live :class:`paramem.server.config.ServerConfig` (or mock with
            ``adapter_dir``, ``key_metadata_path``, ``paths.data``, and
            ``consolidation.mode``).
        store: Optional live :class:`paramem.memory.store.MemoryStore`.  When
            supplied, augments tier enumeration with in-memory registry data.
        daily_loadable: Whether the daily age identity is loadable.  When
            ``False``, ``undecryptable`` entries with the "daily identity not
            loaded" detail are not counted as corruption failures.  When
            ``True``, any decrypt failure is a real failure.

    Returns:
        :class:`IntegrityReport` with all check results and the aggregated
        ``ok`` flag.
    """
    adapter_dir = Path(config.adapter_dir)
    mode = config.consolidation.mode  # "train" | "simulate"
    data_dir = Path(config.paths.data)

    checks: list[FileCheck] = []

    # -----------------------------------------------------------------------
    # Build the set of tiers to check.
    # Main tiers + any interim dirs discovered on disk or in the store.
    # -----------------------------------------------------------------------
    tiers_to_check: list[tuple[str, Path | None, str]] = []
    # (tier_name, slot_root_for_manifest, "main"|"interim")
    for tier in _MAIN_TIERS:
        tiers_to_check.append((tier, adapter_dir / tier, "main"))

    # Interim adapters are episodic-only: day-by-day session slots that
    # collapse into the main tiers on consolidation. (procedural/semantic
    # interim_* adapter slots do not exist; any procedural/interim_* dir holds
    # training debris — epoch_log/progress — not an adapter, and is ignored.)
    from paramem.memory.interim_adapter import iter_interim_dirs

    for interim_name, interim_dir in iter_interim_dirs(adapter_dir):
        tiers_to_check.append((interim_name, interim_dir, "interim"))

    # Also add any tiers from the live store not yet on disk
    if store is not None:
        for store_tier in store.tiers_with_registry():
            if not any(t == store_tier for t, _, _ in tiers_to_check):
                from paramem.memory.interim_adapter import adapter_slot_root_for_name

                slot_root = adapter_slot_root_for_name(adapter_dir, store_tier)
                kind = "interim" if "interim" in store_tier else "main"
                tiers_to_check.append((store_tier, slot_root, kind))

    # -----------------------------------------------------------------------
    # Per-tier checks
    # -----------------------------------------------------------------------
    # Track which registry loads succeeded (for cross-consistency checks).
    registry_ok_keys: dict[str, list[str]] = {}  # tier -> active_keys list
    simhash_ok_keys: dict[str, list[str]] = {}  # tier -> simhash keys list

    for tier_name, tier_root, tier_kind in tiers_to_check:
        tier_root = Path(tier_root)

        # --- Determine if this tier is "committed" (has any data) ---
        # A partial interim slot (dir present but registry absent) is skipped.
        reg_path = tier_root / "indexed_key_registry.json"
        simhash_path = tier_root / "simhash_registry.json"
        graph_path = tier_root / "graph.json"

        # Skip entirely-absent tiers (no dir at all or no registry signal).
        if not tier_root.exists():
            # Absent interim → skipped (not a failure)
            # Absent main tier (e.g. semantic on a fresh install) → skipped
            checks.append(FileCheck(str(reg_path), "registry", tier_name, _SKIPPED, ""))
            continue

        # Partial interim slot: dir present but no registry file → skipped
        if tier_kind == "interim" and not reg_path.exists():
            checks.append(
                FileCheck(str(reg_path), "registry", tier_name, _SKIPPED, "partial interim slot")
            )
            continue

        # --- Registry check ---
        # _check_registry returns (FileCheck, active_keys | None); reuse the
        # parsed payload for cross-consistency so the file is read only once.
        reg_check, reg_active_keys = _check_registry(reg_path, tier_name)
        checks.append(reg_check)

        # Determine whether this tier has active keys (drives required/optional logic).
        has_keys = False
        if reg_check.status == _OK and reg_active_keys is not None:
            has_keys = len(reg_active_keys) > 0
            registry_ok_keys[tier_name] = reg_active_keys

        # Empty registry (zero active keys) → simhash + manifest optional for this tier.
        # Non-existent registry for a main tier → skipped (fresh install or cleared tier).
        if not reg_path.exists():
            # Fresh tier — skip simhash and graph too
            checks.append(FileCheck(str(simhash_path), "simhash", tier_name, _SKIPPED, ""))
            continue

        # --- SimHash check ---
        # _check_simhash returns (FileCheck, simhash_dict | None); reuse the
        # parsed dict for cross-consistency so the file is read only once.
        if not has_keys:
            # Registry loaded ok but is empty → simhash is optional
            simhash_check: FileCheck = FileCheck(
                str(simhash_path), "simhash", tier_name, _SKIPPED, "empty registry"
            )
            simhash_payload: dict | None = None
        elif mode == "train":
            # train mode: simhash required when registry has keys
            if not simhash_path.exists():
                simhash_check = FileCheck(
                    str(simhash_path),
                    "simhash",
                    tier_name,
                    _MISSING,
                    "simhash required in train mode",
                )
                simhash_payload = None
            else:
                simhash_check, simhash_payload = _check_simhash(simhash_path, tier_name)
        else:
            # simulate mode: simhash is also stored alongside graph
            simhash_check, simhash_payload = _check_simhash(simhash_path, tier_name)
        checks.append(simhash_check)

        if simhash_check.status == _OK and simhash_payload is not None:
            simhash_ok_keys[tier_name] = list(simhash_payload.keys())

        # --- Graph check ---
        if mode == "simulate":
            # simulate: graph required for committed tiers (those with a registry)
            if has_keys:
                graph_check = _check_graph(graph_path, tier_name)
            else:
                graph_check = FileCheck(
                    str(graph_path), "graph", tier_name, _SKIPPED, "empty registry"
                )
            checks.append(graph_check)
        else:
            # train: graph is optional / skipped
            _skip_detail = "not required in train mode"
            checks.append(FileCheck(str(graph_path), "graph", tier_name, _SKIPPED, _skip_detail))

        # --- Manifest check (train mode only, live weight slot) ---
        if mode == "train" and has_keys:
            # Find the live weight slot dir (has meta.json)
            live_slot = _find_live_slot_for_tier(adapter_dir, tier_name)
            if live_slot is None:
                # No slot dir found — manifest missing
                meta_path = adapter_dir / tier_name / "meta.json"
                checks.append(
                    FileCheck(str(meta_path), "manifest", tier_name, _MISSING, "no weight slot")
                )
            else:
                manifest_check = _check_manifest(live_slot, tier_name)
                checks.append(manifest_check)
        elif mode == "simulate":
            # simulate: manifest is optional (no weight slot expected)
            live_slot = _find_live_slot_for_tier(adapter_dir, tier_name)
            if live_slot is not None:
                meta_path = live_slot / "meta.json"
                checks.append(
                    FileCheck(str(meta_path), "manifest", tier_name, _SKIPPED, "simulate mode")
                )
            # else: no slot, no check needed

    # -----------------------------------------------------------------------
    # Common files (always optional — fresh installs lack them)
    # -----------------------------------------------------------------------
    key_metadata_path = Path(config.key_metadata_path)
    speaker_profiles_path = data_dir / "speaker_profiles.json"
    observed_languages_path = data_dir / "observed_languages.json"
    backup_state_path = data_dir / "state" / "backup.json"

    # _check_common_file returns (FileCheck, parsed_dict | None); reuse the
    # parsed dict for cross-consistency so the file is read only once.
    key_metadata_check, key_metadata_parsed = _check_common_file(key_metadata_path, "key_metadata")
    checks.append(key_metadata_check)
    checks.append(_check_common_file(speaker_profiles_path, "common")[0])
    checks.append(_check_common_file(observed_languages_path, "common")[0])
    checks.append(_check_common_file(backup_state_path, "common")[0])

    # -----------------------------------------------------------------------
    # Cross-consistency checks
    # -----------------------------------------------------------------------
    # Only for tiers whose registry loaded "ok".
    for tier_name, reg_keys in registry_ok_keys.items():
        sh_keys = simhash_ok_keys.get(tier_name, [])
        reg_set = set(reg_keys)
        sh_set = set(sh_keys)

        # Keys in registry but not in simhash
        missing_from_sh = sorted(reg_set - sh_set)
        if missing_from_sh:
            sample = missing_from_sh[:10]
            detail = f"registry keys without simhash fingerprint: {sample}"
            checks.append(
                FileCheck(
                    str(adapter_dir / tier_name / "simhash_registry.json"),
                    "simhash",
                    tier_name,
                    _INCONSISTENT,
                    detail,
                )
            )

        # Keys in simhash but not in registry (orphan fingerprints)
        orphan_in_sh = sorted(sh_set - reg_set)
        if orphan_in_sh:
            sample = orphan_in_sh[:10]
            detail = f"simhash keys absent from registry: {sample}"
            checks.append(
                FileCheck(
                    str(adapter_dir / tier_name / "simhash_registry.json"),
                    "simhash",
                    tier_name,
                    _INCONSISTENT,
                    detail,
                )
            )

    # Cross-consistency: key_metadata orphans
    # A key in key_metadata["keys"] whose tier_for_active_key() returns None.
    # Uses the already-parsed payload from _check_common_file — no second read.
    if store is not None and key_metadata_check.status == _OK and key_metadata_parsed is not None:
        orphan_keys: list[str] = []
        for key in key_metadata_parsed.get("keys", {}):
            if store.tier_for_active_key(key) is None:
                orphan_keys.append(key)
        if orphan_keys:
            sample = sorted(orphan_keys)[:10]
            checks.append(
                FileCheck(
                    str(key_metadata_path),
                    "key_metadata",
                    "common",
                    _INCONSISTENT,
                    f"key_metadata keys not in any active registry: {sample}",
                )
            )

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    failures: list[FileCheck] = []
    for check in checks:
        if check.status not in (_OK, _SKIPPED):
            # Distinguish no-key undecryptable from real corruption.
            if _is_no_key_check(check) and not daily_loadable:
                # No-key undecryptable when daily identity not loaded — not
                # a corruption failure; caller decides what to do with it.
                continue
            failures.append(check)

    return IntegrityReport(ok=not failures, checks=checks, failures=failures)
