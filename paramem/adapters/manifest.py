"""Per-adapter meta.json schema and live-slot resolver.

Schema version: MANIFEST_SCHEMA_VERSION = 2.

Schema history:
  * v1: original schema (no ``window_stamp``).
  * v2: adds ``window_stamp`` — the cadence-window the slot represents.
    Set at write time by the producer (interim or full-cycle path) and
    used by the Phase 4 full-cycle gate to decide "have we already
    consolidated the current window?" via stamp identity comparison.

Forward-compat / auto-upgrade: ``_dict_to_manifest`` accepts v1 manifests
on read and treats absent ``window_stamp`` as the empty string. Empty
``window_stamp`` on the canonical main ``episodic`` slot is interpreted
by the gate as "unknown window — first full cycle is due", so v1 slots
naturally trigger a re-consolidation that overwrites them with v2 on
first run. ``synthesized`` retains the same forward-compat default of
``False`` when absent.

On-disk layout
--------------
Every adapter save produces a timestamped slot directory::

    data/ha/adapters/episodic/20260421-041237/
        meta.json
        adapter_config.json
        adapter_model.safetensors
        keyed_pairs.json          # optional

``UNKNOWN`` sentinel
--------------------
Single module-level constant used on ``str | int`` union fields when the
value cannot be determined (migration paths, unpinned loads).  Startup
validator consults ``AdapterManifest.synthesized`` to pick severity:

* ``synthesized=True`` + UNKNOWN → yellow (acceptable transient state).
* ``synthesized=False`` + UNKNOWN → red (build_manifest_for failed; surface
  loudly).

References
----------
- Resolved Decision 17: per-adapter manifest schema distinct from ArtifactMeta.
- Resolved Decision 31: live-slot by registry_sha256 match, not pointer file.
- Plan §1.2, §1.3, §1.4, §1.5, §2.1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION: int = 2
UNKNOWN: Final[str] = "unknown"

_MANIFEST_FILENAME = "meta.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseModelFingerprint:
    """Fingerprint of the base language model.

    Attributes:
        repo: HuggingFace model ID (``model.config._name_or_path``) or
            ``UNKNOWN`` for unpinned/local models.
        sha: Commit hash (``model.config._commit_hash``) or ``UNKNOWN``
            when not pinned via ``revision=``.
        hash: ``"sha256:<hex>"`` of the sorted-parameter weight bytes, or
            ``UNKNOWN`` when caching was unavailable.
    """

    repo: str
    sha: str
    hash: str


@dataclass(frozen=True)
class TokenizerFingerprint:
    """Fingerprint of the tokenizer used during training.

    Attributes:
        name_or_path: ``tokenizer.name_or_path`` or ``UNKNOWN``.
        vocab_size: ``len(tokenizer)`` as int, or ``UNKNOWN`` as string when
            unavailable.
        merges_hash: SHA-256 hex of ``tokenizer.json`` bytes, or ``UNKNOWN``.
    """

    name_or_path: str
    vocab_size: int | str  # int or UNKNOWN
    merges_hash: str


@dataclass(frozen=True)
class LoRAShape:
    """LoRA adapter hyper-parameters captured at training time.

    Attributes:
        rank: LoRA rank ``r``.
        alpha: LoRA scaling ``lora_alpha``.
        dropout: ``lora_dropout`` probability.
        target_modules: Tuple of module names (e.g. ``("q_proj", "v_proj")``).
    """

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class AdapterManifest:
    """Immutable per-adapter manifest written alongside every saved adapter.

    Attributes:
        schema_version: Always ``MANIFEST_SCHEMA_VERSION`` (currently 2).
        name: Adapter name string (e.g. ``"episodic"``).
        trained_at: ISO-8601 UTC timestamp (``"YYYY-MM-DDTHH:MM:SSZ"``).
        window_stamp: ``"YYYYMMDDTHHMM"`` cadence-window this slot represents.
            For interim slots (``episodic_interim_<X>``) it is the
            refresh-cadence boundary stamp (same value as the adapter-name
            suffix). For main full-cycle slots it is the full-consolidation
            boundary stamp. The slot's training represents this window: two
            slots with the same ``window_stamp`` were produced by cycles in
            the same cadence boundary. Empty string for legacy v1 manifests
            (auto-upgraded on read) and synthesized fallbacks where the
            window is unknown.
        base_model: Base model fingerprint.
        tokenizer: Tokenizer fingerprint.
        lora: LoRA shape.
        registry_sha256: SHA-256 hex of ``indexed_key_registry.json`` at
            training time; empty string when none; ``UNKNOWN`` for migrated.
        keyed_pairs_sha256: SHA-256 hex of ``keyed_pairs.json``; empty when
            none; ``UNKNOWN`` for migrated.
        key_count: Number of indexed keys in this adapter, or ``UNKNOWN``.
        synthesized: ``True`` **only** for migration-script output.  Drives
            UNKNOWN severity: synthesized + UNKNOWN → yellow; fresh +
            UNKNOWN → red.  Defaults to ``False`` when absent from on-disk
            JSON (forward-compat).
    """

    schema_version: int
    name: str
    trained_at: str
    base_model: BaseModelFingerprint
    tokenizer: TokenizerFingerprint
    lora: LoRAShape
    registry_sha256: str
    keyed_pairs_sha256: str
    key_count: int | str  # int or UNKNOWN
    synthesized: bool = False
    window_stamp: str = ""  # cadence-window stamp; "" = legacy / unknown


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ManifestError(Exception):
    """Base class for all manifest errors."""


class ManifestNotFoundError(ManifestError):
    """``meta.json`` does not exist in the given slot directory."""


class ManifestSchemaError(ManifestError):
    """``meta.json`` could be read but failed schema validation."""


class ManifestFingerprintMismatchError(ManifestError):
    """A manifest field value disagrees with the live runtime state.

    Attributes:
        reason: Human-readable explanation.
        field: Name of the mismatching field, or ``None`` when unspecified.
    """

    def __init__(self, reason: str, field: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.field = field


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _manifest_to_dict(manifest: AdapterManifest) -> dict:
    """Convert a manifest to a JSON-serialisable dict."""
    d = asdict(manifest)
    # tuple is not JSON-serialisable; convert to list
    d["lora"]["target_modules"] = list(manifest.lora.target_modules)
    return d


def _dict_to_manifest(d: dict) -> AdapterManifest:
    """Parse a raw dict from JSON into an AdapterManifest.

    Schema-version handling:
      * v1 (legacy): auto-upgraded in-memory by defaulting ``window_stamp``
        to ``""``. The on-disk file is left untouched until the next save,
        which writes v2.  Empty ``window_stamp`` is interpreted by the
        Phase 4 gate as "unknown window — first full cycle is due".
      * v2 (current): ``window_stamp`` required.
      * Newer-than-current: rejected with ManifestSchemaError so callers
        do not silently downgrade.

    Raises ManifestSchemaError for missing required fields or unsupported
    schema versions.
    """
    required_top = [
        "schema_version",
        "name",
        "trained_at",
        "base_model",
        "tokenizer",
        "lora",
        "registry_sha256",
        "keyed_pairs_sha256",
        "key_count",
    ]
    for field in required_top:
        if field not in d:
            raise ManifestSchemaError(f"Missing required field: {field!r}")

    schema = d["schema_version"]
    if not isinstance(schema, int):
        raise ManifestSchemaError(f"schema_version must be int, got {type(schema)!r}")
    if schema > MANIFEST_SCHEMA_VERSION:
        raise ManifestSchemaError(
            f"schema_version={schema} is newer than supported "
            f"({MANIFEST_SCHEMA_VERSION}); refusing to downgrade silently."
        )

    bm = d["base_model"]
    for f in ("repo", "sha", "hash"):
        if f not in bm:
            raise ManifestSchemaError(f"Missing base_model.{f}")
    base_model = BaseModelFingerprint(repo=bm["repo"], sha=bm["sha"], hash=bm["hash"])

    tok = d["tokenizer"]
    for f in ("name_or_path", "vocab_size", "merges_hash"):
        if f not in tok:
            raise ManifestSchemaError(f"Missing tokenizer.{f}")
    tokenizer = TokenizerFingerprint(
        name_or_path=tok["name_or_path"],
        vocab_size=tok["vocab_size"],
        merges_hash=tok["merges_hash"],
    )

    lo = d["lora"]
    for f in ("rank", "alpha", "dropout", "target_modules"):
        if f not in lo:
            raise ManifestSchemaError(f"Missing lora.{f}")
    lora = LoRAShape(
        rank=lo["rank"],
        alpha=lo["alpha"],
        dropout=lo["dropout"],
        target_modules=tuple(lo["target_modules"]),
    )

    # v1 → v2 auto-upgrade: window_stamp absent in v1; default to "".
    window_stamp = d.get("window_stamp", "")
    if not isinstance(window_stamp, str):
        raise ManifestSchemaError(f"window_stamp must be str, got {type(window_stamp)!r}")

    return AdapterManifest(
        schema_version=d["schema_version"],
        name=d["name"],
        trained_at=d["trained_at"],
        window_stamp=window_stamp,
        base_model=base_model,
        tokenizer=tokenizer,
        lora=lora,
        registry_sha256=d["registry_sha256"],
        keyed_pairs_sha256=d["keyed_pairs_sha256"],
        key_count=d["key_count"],
        synthesized=d.get("synthesized", False),
    )


# ---------------------------------------------------------------------------
# Public I/O
# ---------------------------------------------------------------------------


def write_manifest(slot: Path, manifest: AdapterManifest) -> Path:
    """Serialize *manifest* to ``<slot>/meta.json``.

    Idempotent — overwrites any existing ``meta.json``.  Raises
    ``OSError`` if *slot* does not exist.

    Args:
        slot: Slot directory (must already exist).
        manifest: Manifest to serialise.

    Returns:
        Absolute path of the written ``meta.json``.
    """
    if not slot.exists():
        raise OSError(f"Slot directory does not exist: {slot}")
    dest = slot / _MANIFEST_FILENAME
    payload = json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True).encode("utf-8")
    from paramem.backup.encryption import _atomic_write_bytes

    _atomic_write_bytes(dest, payload)
    return dest


def read_manifest(slot: Path) -> AdapterManifest:
    """Read and validate ``<slot>/meta.json``.

    Args:
        slot: Slot directory containing ``meta.json``.

    Returns:
        Parsed :class:`AdapterManifest`.

    Raises:
        ManifestNotFoundError: ``meta.json`` absent from *slot*.
        ManifestSchemaError: File present but fails JSON parse or schema.
    """
    path = slot / _MANIFEST_FILENAME
    if not path.exists():
        raise ManifestNotFoundError(f"meta.json not found in slot: {slot}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestSchemaError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestSchemaError(f"meta.json root must be a JSON object: {path}")
    return _dict_to_manifest(data)


# ---------------------------------------------------------------------------
# Slot resolution
# ---------------------------------------------------------------------------


def _is_slot_name(name: str) -> bool:
    """Return True when *name* looks like a ``YYYYMMDD-HHMMSS`` timestamp."""
    if len(name) != 15:
        return False
    return name[:8].isdigit() and name[8] == "-" and name[9:].isdigit()


def _slot_mtime(slot: Path) -> float:
    """Return the directory mtime as a float for newest-wins ordering."""
    try:
        return slot.stat().st_mtime
    except OSError:
        return 0.0


def find_live_slot(adapter_kind_dir: Path, live_registry_sha256: str) -> Path | None:
    """Return the slot whose ``meta.registry_sha256`` matches *live_registry_sha256*.

    Scans *adapter_kind_dir* for non-hidden subdirectories with a readable
    ``meta.json``.  ``.pending`` and all other dot-prefixed entries are
    skipped.  Unreadable manifests produce a WARN log and are skipped.

    When multiple slots match (e.g. two identical consecutive saves), the
    newest (highest ``st_mtime``) wins.

    An empty *live_registry_sha256* matches slots whose
    ``meta.registry_sha256`` is also empty — this is the fresh-install /
    experiment path where no registry exists yet.

    Args:
        adapter_kind_dir: Directory scoped to a single adapter kind (e.g.
            ``data/ha/adapters/episodic/``).
        live_registry_sha256: SHA-256 hex of the current on-disk registry,
            or ``""`` for a fresh install.

    Returns:
        Path to the best matching slot, or ``None`` when no match is found.
    """
    if not adapter_kind_dir.is_dir():
        return None

    candidates: list[Path] = []
    for entry in adapter_kind_dir.iterdir():
        if entry.name.startswith("."):
            continue  # skip .pending and other hidden entries
        if not entry.is_dir():
            continue
        try:
            manifest = read_manifest(entry)
        except ManifestNotFoundError:
            # Slot without meta.json — skip silently (may be a non-slot subdir)
            continue
        except ManifestSchemaError as exc:
            logger.warning("find_live_slot: skipping unreadable meta.json in %s: %s", entry, exc)
            continue

        if manifest.registry_sha256 == live_registry_sha256:
            candidates.append(entry)

    if not candidates:
        return None

    return max(candidates, key=_slot_mtime)


def resolve_adapter_slot(base_dir: Path, adapter_name: str, live_hash: str) -> Path | None:
    """Resolve the live adapter slot, handling both pre- and post-migration layouts.

    Tries two layouts in order:

    1. ``find_live_slot(base_dir, live_hash)`` — new flat layout where
       ``base_dir`` is already scoped to a single adapter kind and slots live
       directly under it.
    2. ``find_live_slot(base_dir / adapter_name, live_hash)`` — legacy nested
       layout (pre-migration) where PEFT wrote ``base_dir/<name>/<ts>/``.

    SCOPING CONTRACT: *base_dir* **must** already be scoped to a single
    adapter-kind directory.  Valid examples::

        data/ha/adapters/episodic     # server
        phase_dir / "adapter"         # experiments

    Never pass a run root that contains multiple adapter dirs or sibling
    state files — the newest-wins tie-break is scoped to one adapter only.

    Args:
        base_dir: Adapter-kind directory (must be scoped, not a run root).
        adapter_name: Name of the adapter (e.g. ``"episodic"``).
        live_hash: Registry SHA-256 for hash-match, or ``""`` for empty.

    Returns:
        Path to the matched slot, or ``None`` when no slot matches either
        layout.
    """
    # Try new flat layout first
    slot = find_live_slot(base_dir, live_hash)
    if slot is not None:
        return slot
    # Fall back to legacy nested layout (pre-migration)
    return find_live_slot(base_dir / adapter_name, live_hash)


# ---------------------------------------------------------------------------
# Private helpers for fast base-model hashing
# ---------------------------------------------------------------------------


def _resolve_base_safetensors(repo: str, commit_sha: str) -> "list[Path] | None":
    """Resolve base-model ``*.safetensors`` paths in canonical order, or None.

    For a local path (an existing directory on disk) the function returns a
    sorted list of ``*.safetensors`` files found in that directory.
    ``commit_sha`` may be ``UNKNOWN`` — file existence is the only requirement.

    For an HF Hub repo ID the function resolves shards via
    ``huggingface_hub.try_to_load_from_cache``.  Shard order comes from the
    ``weight_map`` values in ``model.safetensors.index.json`` (preserving
    first-seen insertion order, deduplicated).  Falls back to a single-file
    ``model.safetensors`` when no index is found.

    Returns ``None`` on any resolution failure (missing cache, unresolved
    shards, unexpected errors).

    Args:
        repo: HuggingFace model ID or local directory path.
        commit_sha: Commit hash or ``UNKNOWN``.  Used as ``revision`` for
            HF Hub lookups; ignored for local-path repos.

    Returns:
        Ordered list of resolved ``Path`` objects, or ``None`` when
        resolution fails.
    """
    local = Path(repo)
    if local.is_dir():
        files = sorted(local.glob("*.safetensors"))
        return files if files else None

    # HF Hub path
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST

        revision = commit_sha if commit_sha != UNKNOWN else None

        # Try multi-shard index first
        index_path = try_to_load_from_cache(
            repo,
            "model.safetensors.index.json",
            revision=revision,
        )
        index_found = (
            index_path is not None
            and index_path is not _CACHED_NO_EXIST
            and Path(index_path).exists()
        )
        if index_found:
            index_data = json.loads(Path(index_path).read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            # Build ordered, deduplicated shard list from weight_map values
            seen: set[str] = set()
            shard_names: list[str] = []
            for shard in weight_map.values():
                if shard not in seen:
                    seen.add(shard)
                    shard_names.append(shard)
            resolved: list[Path] = []
            for shard in shard_names:
                shard_path = try_to_load_from_cache(repo, shard, revision=revision)
                if shard_path is None or shard_path is _CACHED_NO_EXIST:
                    return None
                p = Path(shard_path)
                if not p.exists():
                    return None
                resolved.append(p)
            return resolved if resolved else None

        # Try single-file model
        single = try_to_load_from_cache(repo, "model.safetensors", revision=revision)
        if single is not None and single is not _CACHED_NO_EXIST:
            p = Path(single)
            if p.exists():
                return [p]
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("_resolve_base_safetensors: resolution failed for %r: %s", repo, exc)
        return None


def _hash_safetensors_files(paths: "list[Path]") -> str:
    """SHA-256 the concatenation of safetensors files via mmap in given order.

    Opens each file, maps it read-only, feeds 1-MiB chunks into a single
    running ``hashlib.sha256`` digest, then closes the mmap and fd before
    opening the next file.

    Args:
        paths: Ordered list of safetensors file paths to hash.

    Returns:
        ``"sha256:<64-char hex>"`` of the concatenated file bytes.

    Raises:
        OSError: When any file cannot be opened or mmap'd.
    """
    h = hashlib.sha256()
    chunk_size = 1024 * 1024  # 1 MiB
    for p in paths:
        fd = p.open("rb")
        try:
            mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                while True:
                    chunk = mm.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
            finally:
                mm.close()
        finally:
            fd.close()
    return "sha256:" + h.hexdigest()


def _lookup_hash_from_manifests(
    adapter_root: Path,
    repo: str,
    commit_sha: str,
) -> "str | None":
    """Walk *adapter_root* for a slot whose ``base_model`` matches *(repo, commit_sha)*.

    Iterates one level of tier dirs under ``adapter_root`` (e.g.
    ``episodic``, ``semantic``, ``procedural``, ``episodic_interim_*``),
    then one more level for slot dirs, reading each ``meta.json``.  Slots
    inside ``.pending`` are skipped.  Unreadable manifests are skipped with
    a debug log.

    Slots whose ``base_model.hash == UNKNOWN`` are excluded — re-emitting
    ``UNKNOWN`` as a cache hit would permanently lock the cache to
    ``UNKNOWN``.

    When multiple matching slots disagree on ``base_model.hash``, a WARN
    is logged with both values before returning the newest (by
    ``trained_at`` ISO-8601 string comparison).

    Args:
        adapter_root: Root directory containing tier subdirs.
        repo: Expected ``base_model.repo`` to match.
        commit_sha: Expected ``base_model.sha`` to match.  ``UNKNOWN``
            always returns ``None`` (no reliable match key).

    Returns:
        ``"sha256:<hex>"`` from a matching slot, or ``None`` when no
        reliable match is found.
    """
    if commit_sha == UNKNOWN:
        return None
    if not adapter_root.is_dir():
        return None

    matches: list[tuple[str, str]] = []  # (trained_at, base_model_hash)
    for tier_dir in adapter_root.iterdir():
        if tier_dir.name.startswith("."):
            continue
        if not tier_dir.is_dir():
            continue
        for slot_dir in tier_dir.iterdir():
            if slot_dir.name.startswith("."):
                continue
            if not slot_dir.is_dir():
                continue
            try:
                manifest = read_manifest(slot_dir)
            except (ManifestNotFoundError, ManifestSchemaError) as exc:
                logger.debug("_lookup_hash_from_manifests: skipping %s: %s", slot_dir, exc)
                continue
            bm = manifest.base_model
            if bm.repo != repo or bm.sha != commit_sha:
                continue
            if bm.hash == UNKNOWN:
                continue
            matches.append((manifest.trained_at, bm.hash))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0], reverse=True)  # newest trained_at first
    newest_hash = matches[0][1]

    # Warn if multiple distinct hashes found for same (repo, sha)
    distinct_hashes = {h for _, h in matches}
    if len(distinct_hashes) > 1:
        logger.warning(
            "_lookup_hash_from_manifests: disagreeing base_model.hash values for "
            "repo=%r sha=%r — hashes found: %s; using newest (trained_at=%s)",
            repo,
            commit_sha,
            sorted(distinct_hashes),
            matches[0][0],
        )

    return newest_hash


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_manifest_for(
    model,
    tokenizer,
    adapter_name: str,
    *,
    registry_path: "Path | None",
    keyed_pairs_path: "Path | None",
    key_count: "int | None" = None,
    base_model_hash_cache: "dict | None" = None,
    registry_sha256_override: "str | None" = None,
    window_stamp: str = "",
    adapter_root: "Path | None" = None,
) -> AdapterManifest:
    """Build an :class:`AdapterManifest` for a live model/tokenizer.

    All fingerprinting happens here — single provider for every caller in
    the training and server paths.  ``synthesized`` is always ``False``
    (reserved for the migration script).

    The base-model weight hash is computed using three escalating strategies
    (cheapest first):

    1. **In-memory cache** (``base_model_hash_cache``): keyed by ``id(model)``,
       populated on first call.  Resets on every process restart.
    2. **Manifest read-back** (warm path): when ``adapter_root`` is given and
       ``base_sha != UNKNOWN``, scans existing slot ``meta.json`` files under
       *adapter_root* for a matching ``(base_model.repo, base_model.sha)``
       entry.  The slots are already on disk from prior consolidations, so no
       hashing work is needed.  ``UNKNOWN`` entries are ignored (re-emitting
       UNKNOWN would permanently lock the cache).
    3. **Source-safetensors mmap-hash** (cold path): resolves the base-model
       ``*.safetensors`` files from disk (local path) or the HF Hub cache
       (via ``try_to_load_from_cache``), then SHA-256s the concatenation via
       mmap.  Much faster than the old in-memory ``state_dict`` walk (~8.5 min
       for Mistral 7B) and survives process restarts.

    If all three strategies fail, ``base_hash`` is set to ``UNKNOWN`` and a
    warning is logged.  The old ``state_dict`` walk is not used.

    Args:
        model: A ``PeftModel`` (or base model) with a ``config`` attribute.
        tokenizer: A HuggingFace tokenizer with ``name_or_path`` and
            ``tokenizer.json``/``backend_tokenizer``.
        adapter_name: Name of the adapter being saved.
        registry_path: Path to ``indexed_key_registry.json`` on disk, or
            ``None`` when the registry has not been written yet.  Ignored
            when *registry_sha256_override* is provided.
        keyed_pairs_path: Path to ``keyed_pairs.json``, or ``None``.
        key_count: Number of indexed keys; inferred from
            ``keyed_pairs_path`` when ``None``.
        base_model_hash_cache: Optional mutable dict used to cache the
            base-model weight hash.  Caller owns it; pass ``_state`` on the
            server path and a local dict in experiments.
        registry_sha256_override: When provided, used directly as
            ``registry_sha256`` instead of reading *registry_path*.  Used
            by the I5 reorder path (§2.5) where the payload bytes have
            already been hashed before writing to disk.
        window_stamp: ``"YYYYMMDDTHHMM"`` cadence-window the slot represents
            (see ``AdapterManifest.window_stamp``).  Empty string when the
            caller cannot determine the window — e.g. ad-hoc experiment
            paths.  Production writers pass the floored cadence boundary:
            ``current_interim_stamp(refresh_cadence)`` for interim slots,
            ``current_full_consolidation_stamp(period)`` for main slots.
        adapter_root: Root directory of the adapter store (single path,
            not a sequence).  When provided, enables the manifest read-back
            warm path: existing ``meta.json`` files under this root are
            scanned for a matching ``(base_model.repo, base_model.sha)``
            entry before attempting the file-hash cold path.  Pass
            ``self.output_dir`` from the consolidation loop or
            ``Path(config.adapter_dir)`` from the migration helper.
            Experiment callers should leave this ``None`` — they skip
            read-back and get the file-hash speedup instead.

    Returns:
        A fully-populated :class:`AdapterManifest` with ``synthesized=False``.
    """

    # --- trained_at ---
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- base_model fingerprint ---
    config = getattr(model, "config", None)
    base_repo = UNKNOWN
    base_sha = UNKNOWN
    base_hash = UNKNOWN

    if config is not None:
        base_repo = getattr(config, "_name_or_path", UNKNOWN) or UNKNOWN
        base_sha = getattr(config, "_commit_hash", UNKNOWN) or UNKNOWN

    # Weight hash — three strategies (cheapest first):
    # 1. In-memory cache hit (amortises within one process lifetime).
    # 2. Manifest read-back: scan existing meta.json files on disk.
    # 3. Source safetensors mmap-hash (cold path; much faster than state_dict walk).
    cache_key = id(model)
    if base_model_hash_cache is not None and cache_key in base_model_hash_cache:
        # Strategy 1: in-memory cache hit
        base_hash = base_model_hash_cache[cache_key]
    else:
        base_hash = UNKNOWN

        # Strategy 2: manifest read-back (warm path)
        if adapter_root is not None and base_sha != UNKNOWN:
            read_back = _lookup_hash_from_manifests(adapter_root, base_repo, base_sha)
            if read_back is not None:
                base_hash = read_back
                logger.debug(
                    "build_manifest_for: base_model hash from manifest read-back (%s)",
                    base_hash[:20] + "…",
                )

        # Strategy 3: source safetensors mmap-hash (cold path)
        if base_hash == UNKNOWN:
            try:
                paths = _resolve_base_safetensors(base_repo, base_sha)
                if paths:
                    base_hash = _hash_safetensors_files(paths)
                    logger.debug(
                        "build_manifest_for: base_model hash from safetensors files (%s)",
                        base_hash[:20] + "…",
                    )
                else:
                    logger.warning(
                        "build_manifest_for: could not resolve safetensors files for "
                        "repo=%r sha=%r — base_model.hash will be UNKNOWN",
                        base_repo,
                        base_sha,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "build_manifest_for: safetensors hash failed for repo=%r: %s",
                    base_repo,
                    exc,
                )

        if base_model_hash_cache is not None:
            base_model_hash_cache[cache_key] = base_hash

    base_model_fp = BaseModelFingerprint(repo=base_repo, sha=base_sha, hash=base_hash)

    # --- tokenizer fingerprint ---
    tok_name = getattr(tokenizer, "name_or_path", UNKNOWN) or UNKNOWN
    try:
        vocab_size: int | str = len(tokenizer)
    except Exception:  # noqa: BLE001
        vocab_size = UNKNOWN

    merges_hash = UNKNOWN
    try:
        # Try fast path: read tokenizer.json from disk
        tok_path = getattr(tokenizer, "vocab_file", None)
        tok_json_path = None
        if tok_path:
            candidate = Path(tok_path).parent / "tokenizer.json"
            if candidate.exists():
                tok_json_path = candidate
        if tok_json_path is None:
            # Try name_or_path directory
            if tok_name != UNKNOWN:
                candidate = Path(tok_name) / "tokenizer.json"
                if candidate.exists():
                    tok_json_path = candidate
        if tok_json_path is not None:
            merges_hash = hashlib.sha256(tok_json_path.read_bytes()).hexdigest()
        else:
            # Fallback: serialize via backend_tokenizer
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is not None:
                merges_hash = hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest()
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_manifest_for: could not hash tokenizer: %s", exc)
        merges_hash = UNKNOWN

    tokenizer_fp = TokenizerFingerprint(
        name_or_path=tok_name,
        vocab_size=vocab_size,
        merges_hash=merges_hash,
    )

    # --- LoRA shape ---
    lora_rank = 0
    lora_alpha = 0
    lora_dropout = 0.0
    lora_targets: tuple[str, ...] = ()
    try:
        peft_cfg = model.peft_config.get(adapter_name) if hasattr(model, "peft_config") else None
        if peft_cfg is not None:
            lora_rank = int(getattr(peft_cfg, "r", 0))
            lora_alpha = int(getattr(peft_cfg, "lora_alpha", 0))
            lora_dropout = float(getattr(peft_cfg, "lora_dropout", 0.0))
            targets = getattr(peft_cfg, "target_modules", ())
            lora_targets = tuple(sorted(targets)) if targets else ()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "build_manifest_for: could not read LoRA config for %s: %s", adapter_name, exc
        )

    lora = LoRAShape(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_targets,
    )

    # --- registry_sha256 ---
    # Hash the PLAINTEXT content, not the on-disk bytes. age re-encrypts with a
    # fresh content key on every write, so a ciphertext-based hash would change
    # on every re-encrypt and break live-slot drift detection.
    # read_maybe_encrypted unwraps the age envelope when present and returns
    # the original bytes otherwise.
    if registry_sha256_override is not None:
        registry_sha256 = registry_sha256_override
    elif registry_path is not None and registry_path.exists():
        try:
            from paramem.backup.encryption import read_maybe_encrypted

            registry_sha256 = hashlib.sha256(read_maybe_encrypted(registry_path)).hexdigest()
        except (OSError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "build_manifest_for: could not hash registry at %s: %s", registry_path, exc
            )
            registry_sha256 = UNKNOWN
    else:
        registry_sha256 = ""

    # --- keyed_pairs_sha256 + key_count ---
    keyed_pairs_sha256: str = ""
    resolved_key_count: int | str = key_count if key_count is not None else UNKNOWN

    if keyed_pairs_path is not None and keyed_pairs_path.exists():
        try:
            # Hash the PLAINTEXT content — envelope-wrapped keyed_pairs would
            # produce a different hash on every re-encrypt (fresh age content
            # key). See registry_sha256 above for the same reasoning.
            from paramem.backup.encryption import read_maybe_encrypted

            raw = read_maybe_encrypted(keyed_pairs_path)
            keyed_pairs_sha256 = hashlib.sha256(raw).hexdigest()
            if key_count is None:
                try:
                    data = json.loads(raw)
                    resolved_key_count = len(data)
                except (json.JSONDecodeError, TypeError):
                    resolved_key_count = UNKNOWN
        except OSError as exc:
            logger.warning(
                "build_manifest_for: could not hash keyed_pairs at %s: %s",
                keyed_pairs_path,
                exc,
            )
            keyed_pairs_sha256 = UNKNOWN

    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=adapter_name,
        trained_at=trained_at,
        window_stamp=window_stamp,
        base_model=base_model_fp,
        tokenizer=tokenizer_fp,
        lora=lora,
        registry_sha256=registry_sha256,
        keyed_pairs_sha256=keyed_pairs_sha256,
        key_count=resolved_key_count,
        synthesized=False,
    )
