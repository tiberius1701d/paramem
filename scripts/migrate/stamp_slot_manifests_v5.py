#!/usr/bin/env python3
"""One-shot migration: stamp every slot manifest to schema v5.

The v5 :class:`~paramem.adapters.manifest.AdapterManifest` adds a payload
content fingerprint (``payload.kind`` + the plaintext SHA-256 of the slot's
own payload file) and drops the read-side degradation arms earlier schema
versions relied on. ``read_manifest`` becomes current-schema-only: it
refuses every prior-shape ``meta.json`` outright. This script is the one
place that still reads the prior shape — on BOTH artifacts a slot's
manifest is bound against, ``meta.json`` and the tier's own
``indexed_key_registry.json`` — and rewrites them once, offline, so every
subsequent read goes through the ordinary, current-schema-only package
primitives.

The content a weight slot carries — the key triples encoded in its
payload — does not change across this release; only the METADATA moves.
Registry bytes are re-serialized through the same canonical writer the
live server uses (:meth:`~paramem.training.key_registry.KeyRegistry.save_bytes`)
so a registry write only actually lands when that re-serialization changes
the bytes — a clean re-run is a true no-op, not a fresh re-encrypt of
unchanged content. A pre-migration registry — one whose ``"stale"`` section
is still a dict of per-id records (with or without ``stale_since`` or a
withheld-id fingerprint), or one predating that section entirely — is read
directly as JSON (never through :class:`KeyRegistry`, which refuses that
shape outright) and rewritten with ``"stale"`` as a bare sorted array of ids
and no fingerprint on any withheld id, before being handed to
:meth:`KeyRegistry.load_from_bytes` for validation; see
:func:`_migrate_registry_bytes`.

What this script walks: every tier root :func:`~paramem.memory.interim_adapter.iter_tier_roots`
yields (the three main tiers plus every interim slot) and every donor store
:func:`~paramem.training.donor.iter_donor_stores` finds — the same
enumerations boot itself uses, so migration and boot cannot disagree about
what a slot is.

Per tier, one pass, registry last:

1. A slot directory missing any of ``required_slot_files("train")`` is
   partial-trained scratch (the same predicate
   :func:`~paramem.backup.integrity.cleanup_partial_slots` applies to main
   tiers) and is skipped — boot reaps it.
2. A registry file present with ZERO slot candidates is the pre-upgrade
   simulate-venue shape (content, no manifest, by construction). Its
   registry is still walked, in case a later run adds content-rule
   changes, but no manifest is invented — that tier is not carried
   forward; boot leaves it unpublishable loudly on its own.
3. Candidates present: every slot whose recorded ``registry_sha256``
   matches the tier's registry hash from BEFORE this pass, or the hash the
   re-serialization produces, is the tier's bound slot and gets
   ``registry_sha256`` re-stamped to the (possibly re-serialized) digest;
   every other candidate keeps its own recorded ``registry_sha256``
   verbatim. A slot is written whenever its OWN ``schema_version`` is not
   current OR (bound slot only) its recorded ``registry_sha256`` does not
   yet match the tier's final digest — idempotence is binding-inclusive,
   not schema-version-gated: a bound slot already at the current schema
   version but still carrying the pre-rewrite registry digest (e.g. its
   registry was pre-change-shape and this pass just rewrote it) is
   restamped exactly like a schema-outdated one. Bound-slot resolution
   happens BEFORE any write, from the raw on-disk manifests, and each
   rewritten slot's directory mtime is restored afterward —
   ``write_manifest``'s atomic write bumps it, and a bumped mtime could
   silently change which slot :func:`find_live_slot`'s newest-wins
   tie-break resolves to.
4. Candidates present, none matching either digest: the tier was already
   unbound before this script ran (pre-existing damage) — STOP, loudly,
   naming the tier. This script adapts metadata; it does not repair a
   broken binding.

A donor store carries no registry by design (``registry_sha256=""``,
matching the empty-registry match convention) and takes the manifest pass
only. Everything else — an unreadable/malformed ``meta.json``, an
unsupported schema version, an undecryptable payload or registry, a
registry that will not parse as a :class:`~paramem.training.key_registry.KeyRegistry`
— stops the run loudly at the first occurrence, naming the tier. No
partial tolerance, no skip-and-continue: the standing no-auto-heal ruling
governs this migration exactly as it governs boot.

``--dry-run`` is read-only: it reports what would change and writes
nothing (and skips the liveness check below, so it is safe to run
alongside a live server for inspection).

Concurrency guard: refuses to run (without ``--dry-run``) while the
``paramem-server`` user service is active (via
:mod:`paramem.utils.systemctl`) or a recognised training script — or, on a
non-systemd host where the service check degrades, the bare server process
itself — is alive (via ``pgrep`` — the same pattern
``outputs_to_slot_dirs.py`` establishes) — this script mutates the live
adapter tree in place, unlike the scratch-``outputs/`` migrations, so both
a live server (which trains in its own background loop) and a standalone
training script are refused. This refusal is unconditional — there is no
override; stop the server and pause training first.

Usage::

    python scripts/migrate/stamp_slot_manifests_v5.py \\
        --adapter-root data/ha/adapters \\
        [--dry-run] \\
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running directly from the repo root without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyrage  # noqa: E402  (sys.path setup above)

from paramem.adapters.manifest import (  # noqa: E402  (sys.path setup above)
    MANIFEST_SCHEMA_VERSION,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    PayloadFingerprint,
    TokenizerFingerprint,
    iter_slot_candidates,
    write_manifest,
)
from paramem.adapters.slot import payload_filename, required_slot_files  # noqa: E402
from paramem.backup.encryption import read_maybe_encrypted  # noqa: E402
from paramem.backup.hashing import content_sha256_bytes, plaintext_sha256  # noqa: E402
from paramem.memory.interim_adapter import iter_tier_roots  # noqa: E402
from paramem.training.donor import iter_donor_stores  # noqa: E402
from paramem.training.key_registry import KeyRegistry  # noqa: E402
from paramem.utils import systemctl  # noqa: E402

logger = logging.getLogger("stamp_slot_manifests_v5")

_REGISTRY_FILENAME = "indexed_key_registry.json"

# pgrep patterns for a live process this script must refuse to run
# alongside: the standalone training scripts outputs_to_slot_dirs.py already
# checks, PLUS the server process itself. The server patterns are the
# fallback for a non-systemd server: _server_active() degrades to False when
# systemctl is unavailable (best-effort, not a hard refusal), so a server
# started by hand (`python -m paramem.server` / `uvicorn ...`) is caught
# here instead of slipping past both checks.
_LIVE_PROCESS_PGREP_PATTERNS: list[str] = [
    "test4b_",
    "test6_",
    "test8_",
    "test10_",
    "test11_",
    "test13_",
    "paramem.server",
    "uvicorn",
]


class TierMigrationBlocked(RuntimeError):
    """Raised to stop the whole run, loudly, naming one tier.

    Attributes:
        tier: The tier or donor-store label the failure occurred in.
        reason: Human-readable explanation.
    """

    def __init__(self, tier: str, reason: str) -> None:
        super().__init__(f"{tier}: {reason}")
        self.tier = tier
        self.reason = reason


@dataclass(frozen=True)
class _TierResult:
    """One tier's outcome, for the run summary."""

    label: str
    action: str
    migrated_slots: tuple[str, ...] = ()


def _pgrep_alive(patterns: list[str]) -> list[tuple[str, str]]:
    """Return ``(pid, pattern)`` pairs for any alive process matching *patterns*.

    Args:
        patterns: List of ``pgrep -f`` pattern strings.

    Returns:
        List of ``(pid_str, pattern)`` tuples for running matches.
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


def _server_active() -> bool:
    """Return ``True`` when the ``paramem-server`` user service is active.

    Routes through :mod:`paramem.utils.systemctl` — the single
    monkeypatchable ``systemctl --user`` transport boundary every production
    caller uses — rather than shelling out directly.

    Best-effort: when ``systemctl`` is unavailable (non-systemd host, CI
    sandbox) this logs a warning and returns ``False`` rather than blocking
    the run on a check it cannot perform. A non-systemd server process is
    still caught by :func:`_pgrep_alive`'s ``paramem.server``/``uvicorn``
    patterns.
    """
    try:
        result = systemctl.run("is-active", "--quiet", "paramem-server", timeout=5)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("_server_active: could not query systemctl (%s) — assuming inactive", exc)
        return False


def _require_field(d: dict, field: str, *, tier: str, where: str) -> object:
    """Return ``d[field]``, raising :class:`TierMigrationBlocked` when absent."""
    if field not in d:
        raise TierMigrationBlocked(tier, f"{where} missing required field {field!r}")
    return d[field]


def _stamp_slot(
    slot: Path,
    raw: dict,
    *,
    tier: str,
    registry_sha256: str,
) -> AdapterManifest:
    """Build the v5 :class:`AdapterManifest` for one prior-shape ``meta.json``.

    Every fingerprint field is carried over verbatim from *raw* — this
    script performs no content transformation, only a metadata reshape —
    except ``payload``, which is computed fresh from the bytes on disk, and
    ``registry_sha256``, which the caller has already resolved per the
    two-digest bound-slot rule.

    Args:
        slot: The slot directory *raw* was read from.
        raw: The prior-shape ``meta.json`` dict, already schema-checked for
            a supported (<= current) integer ``schema_version``.
        tier: Label for error messages.
        registry_sha256: The value to stamp — either the resolved digest
            (this slot is the tier's bound slot) or *raw*'s own recorded
            value (every other slot), per the caller's own rule.

    Returns:
        A fully-populated v5 :class:`AdapterManifest`.

    Raises:
        TierMigrationBlocked: A required top-level or fingerprint field is
            missing, or the payload file cannot be hashed (undecryptable,
            unreadable). ``name``, ``trained_at`` and ``key_count`` were
            required in every prior schema version (v1-v4); their absence
            means malformed input, not a legitimate omission, so each is a
            hard stop rather than a silent default. ``window_stamp`` is the
            one documented exception (absent in v1 by design) and is
            defaulted, not required, by the caller.
    """
    where = f"meta.json at {slot}"
    name = _require_field(raw, "name", tier=tier, where=where)
    trained_at = _require_field(raw, "trained_at", tier=tier, where=where)
    key_count = _require_field(raw, "key_count", tier=tier, where=where)
    bm = raw.get("base_model") or {}
    tok = raw.get("tokenizer") or {}
    lo = raw.get("lora") or {}
    for field in ("repo", "sha", "hash"):
        _require_field(bm, field, tier=tier, where=f"{where} base_model")
    for field in ("name_or_path", "vocab_size", "merges_hash"):
        _require_field(tok, field, tier=tier, where=f"{where} tokenizer")
    for field in ("rank", "alpha", "dropout", "target_modules"):
        _require_field(lo, field, tier=tier, where=f"{where} lora")

    payload_path = slot / payload_filename("train")
    try:
        digest = plaintext_sha256(payload_path)
    except (RuntimeError, pyrage.DecryptError, OSError) as exc:
        raise TierMigrationBlocked(tier, f"cannot hash payload at {payload_path}: {exc}") from exc

    return AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=name,
        trained_at=trained_at,
        payload=PayloadFingerprint(kind="train", sha256=digest),
        base_model=BaseModelFingerprint(repo=bm["repo"], sha=bm["sha"], hash=bm["hash"]),
        tokenizer=TokenizerFingerprint(
            name_or_path=tok["name_or_path"],
            vocab_size=tok["vocab_size"],
            merges_hash=tok["merges_hash"],
        ),
        lora=LoRAShape(
            rank=lo["rank"],
            alpha=lo["alpha"],
            dropout=lo["dropout"],
            target_modules=tuple(lo.get("target_modules") or []),
        ),
        registry_sha256=registry_sha256,
        key_count=key_count,
        synthesized=bool(raw.get("synthesized", False)),
        # window_stamp: absent in the v1 shape by design (added later) --
        # defaulted to "" (unknown), never a required field, unlike the
        # other four top-level fields above.
        window_stamp=raw.get("window_stamp", ""),
    )


def _migrate_registry_bytes(
    raw_bytes: bytes, *, tier: str, registry_path: Path
) -> tuple[KeyRegistry, bytes]:
    """Produce this tier's registry bytes in the withheld-id-marker shape.

    Reads *raw_bytes* directly as JSON — never through :class:`KeyRegistry`,
    which refuses a pre-migration file outright, which is precisely why no
    reader on the live path carries an old-shape branch. This covers both an
    already-migrated file and a pre-migration one, whose ``"stale"`` section
    is a dict of per-id records (with or without ``stale_since`` or a
    withheld-id fingerprint) or absent entirely. The migrated payload carries
    ``"stale"`` as bare sorted ids — every ``stale_since`` dropped, and every
    fingerprint carried on a withheld id (in its own stale record, or as a
    ``"simhash"`` entry naming it) dropped with it — while everything else
    (active keys, their fingerprints) is carried verbatim. There is no public
    path from a raw dict to a serialized registry carrying a withheld id
    (:meth:`KeyRegistry.stale` no-ops for a key that is not active), so the
    migrated dict is built here as plain JSON, then validated by parsing
    through the one new-shape parser, :meth:`KeyRegistry.load_from_bytes`
    (which also catches an id present in both ``"active_keys"`` and
    ``"stale"``, a non-int fingerprint, or any other shape defect), and
    re-serialized through :meth:`KeyRegistry.save_bytes` to obtain canonical
    bytes. Byte-stability then holds structurally, not by assertion: an
    already-migrated payload re-parses and re-serializes to the same bytes,
    so re-running this transform on a migrated tree is a true no-op.

    Args:
        raw_bytes: The tier's raw, already-decrypted registry bytes.
        tier: Label for error messages.
        registry_path: Used only to name the file in a raised
            :class:`TierMigrationBlocked` — no file is read here.

    Returns:
        ``(registry_obj, canonical_bytes)`` — the parsed registry and the
        canonical migrated bytes ready for the registry-last write.

    Raises:
        TierMigrationBlocked: *raw_bytes* is not valid JSON; is not a dict
            with a list-valued ``"active_keys"`` and a dict-valued
            ``"simhash"``; its ``"stale"`` section is neither a dict of
            records nor a list of string ids; or the migrated payload fails
            :meth:`KeyRegistry.load_from_bytes`'s validation (an id present
            in both ``"active_keys"`` and ``"stale"``, a non-int fingerprint,
            or a ``"simhash"`` entry still naming a withheld id).
    """
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TierMigrationBlocked(
            tier, f"registry at {registry_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("active_keys"), list):
        raise TierMigrationBlocked(
            tier,
            f"registry at {registry_path} is not KeyRegistry-shaped (missing list-valued "
            "'active_keys')",
        )
    simhash_raw = data.get("simhash")
    if not isinstance(simhash_raw, dict):
        raise TierMigrationBlocked(
            tier,
            f"registry at {registry_path} is not KeyRegistry-shaped (missing dict-valued "
            "'simhash')",
        )

    stale_raw = data.get("stale", {})
    if isinstance(stale_raw, list):
        if not all(isinstance(k, str) for k in stale_raw):
            raise TierMigrationBlocked(
                tier, f"registry at {registry_path} 'stale' array contains a non-string id"
            )
        stale_ids = sorted(stale_raw)
    elif isinstance(stale_raw, dict):
        stale_ids = sorted(stale_raw.keys())
    else:
        raise TierMigrationBlocked(
            tier, f"registry at {registry_path} 'stale' is neither a list nor a dict"
        )

    stale_set = set(stale_ids)
    migrated = {
        "active_keys": data["active_keys"],
        "stale": stale_ids,
        "simhash": {k: fp for k, fp in simhash_raw.items() if k not in stale_set},
    }
    migrated_bytes = json.dumps(migrated, indent=2).encode("utf-8")

    try:
        registry_obj = KeyRegistry.load_from_bytes(migrated_bytes, path=registry_path)
    except ValueError as exc:
        raise TierMigrationBlocked(
            tier, f"migrated registry at {registry_path} failed validation: {exc}"
        ) from exc
    return registry_obj, registry_obj.save_bytes()


def _migrate_tier(label: str, tier_root: Path, *, dry_run: bool) -> _TierResult:
    """Migrate one tier root (main tier, interim slot, or donor store).

    Args:
        label: Human-readable tier/store name for logging and errors.
        tier_root: Directory holding this tier's slot candidates and (for a
            memory tier, not a donor store) its own ``indexed_key_registry.json``.
        dry_run: Read-only — compute and log what would change; write
            nothing.

    Returns:
        A :class:`_TierResult` describing the outcome.

    Raises:
        TierMigrationBlocked: Malformed on-disk state, an unsupported
            schema version, an undecryptable artifact, or (candidates
            present) none binding to the tier's registry — see the module
            docstring for the full rule set.
    """
    if not tier_root.is_dir():
        return _TierResult(label, "absent")

    candidates = sorted(iter_slot_candidates(tier_root), key=lambda p: p.name)
    required = required_slot_files("train")
    complete = [c for c in candidates if all((c / f).exists() for f in required)]
    scratch = [c for c in candidates if c not in complete]
    if scratch:
        logger.info(
            "%s: %d partial-trained scratch slot(s) skipped (boot reaps these): %s",
            label,
            len(scratch),
            ", ".join(s.name for s in scratch),
        )

    registry_path = tier_root / _REGISTRY_FILENAME
    registry_exists = registry_path.exists()

    if not candidates:
        # ZERO slot candidates -- the pre-upgrade simulate-venue tier shape
        # (content, no manifest, by construction). See _migrate_registry_only.
        if not registry_exists:
            return _TierResult(label, "untouched")
        return _migrate_registry_only(label, registry_path, dry_run=dry_run)

    if not complete:
        # Candidates ARE present, but every one is partial-trained scratch --
        # excluded from the bound-slot match set (the first tier-shape rule),
        # so this tier already carries no complete slot to bind, exactly the
        # third rule's condition ("candidates present, none binding"). Stops
        # loudly here rather than falling through to a bound-slot resolution
        # that would find nothing to iterate.
        raise TierMigrationBlocked(
            label,
            f"{len(candidates)} candidate slot(s) present, all partial-trained "
            "scratch (excluded from the bound-slot match set) -- tier already "
            "unbound before migration; investigate before migrating",
        )

    # Read every candidate's raw meta.json — schema-checked here, never via
    # the current-schema-only read_manifest.
    raw_metas: dict[Path, dict] = {}
    for slot in complete:
        meta_path = slot / "meta.json"
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TierMigrationBlocked(
                label, f"unreadable meta.json at {meta_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise TierMigrationBlocked(label, f"meta.json at {meta_path} root is not a JSON object")
        schema = raw.get("schema_version")
        if not isinstance(schema, int) or not (1 <= schema <= MANIFEST_SCHEMA_VERSION):
            raise TierMigrationBlocked(
                label, f"meta.json at {meta_path} has unsupported schema_version={schema!r}"
            )
        # registry_sha256 was required in every prior schema version (v1-v4);
        # its absence means malformed input, not a legitimate "no registry"
        # state (that state is recorded as the empty string, still present).
        _require_field(raw, "registry_sha256", tier=label, where=f"meta.json at {meta_path}")
        raw_metas[slot] = raw

    pending = {s: m for s, m in raw_metas.items() if m["schema_version"] != MANIFEST_SCHEMA_VERSION}

    if registry_exists:
        try:
            raw_registry_bytes = read_maybe_encrypted(registry_path)
        except (RuntimeError, pyrage.DecryptError, OSError) as exc:
            raise TierMigrationBlocked(
                label, f"cannot read registry at {registry_path}: {exc}"
            ) from exc
        old_digest = content_sha256_bytes(raw_registry_bytes)
        registry_obj, new_registry_bytes = _migrate_registry_bytes(
            raw_registry_bytes, tier=label, registry_path=registry_path
        )
        new_digest = content_sha256_bytes(new_registry_bytes)
    else:
        # Donor-store convention: no registry file, empty-hash match
        # (manifest.py's documented empty-registry convention).
        raw_registry_bytes = None
        new_registry_bytes = None
        old_digest = ""
        new_digest = None

    def _matches(meta: dict) -> bool:
        rs = meta.get("registry_sha256")
        return rs == old_digest or (new_digest is not None and rs == new_digest)

    bound = {s for s, m in raw_metas.items() if _matches(m)}
    if not bound:
        digests = f"old={old_digest[:12]}…"
        if new_digest is not None:
            digests += f" or new={new_digest[:12]}…"
        raise TierMigrationBlocked(
            label,
            f"{len(raw_metas)} candidate slot(s) present, none carrying registry_sha256 "
            f"matching {digests} — tier already unbound before migration; "
            "investigate before migrating",
        )

    registry_needs_write = (
        registry_exists
        and new_registry_bytes is not None
        and new_registry_bytes != raw_registry_bytes
    )

    # Idempotence is binding-inclusive (module docstring / the sibling
    # plan's Q1): a tier is a no-op iff its registry already parses under
    # the new shape AND its bound slot's manifest already carries THAT
    # registry's digest. A bound slot can carry a stale registry_sha256
    # (still recording old_digest) even when its OWN schema_version is
    # already current -- a registry rewrite (pre-change "stale" shape ->
    # bare sorted ids) changes the tier's digest independently of any
    # slot's schema version, so restamping cannot be gated on
    # schema_version alone. `final_digest` is the digest every bound slot
    # must carry once this pass converges; any bound slot not already
    # carrying it needs a restamp whether or not it is also in `pending`.
    final_digest = new_digest if new_digest is not None else old_digest
    bound_needs_restamp = {s for s in bound if raw_metas[s]["registry_sha256"] != final_digest}
    to_stamp: dict[Path, dict] = dict(pending)
    for slot in bound_needs_restamp:
        to_stamp.setdefault(slot, raw_metas[slot])

    if not to_stamp and not registry_needs_write:
        return _TierResult(label, "already_migrated")

    for slot, raw in to_stamp.items():
        registry_sha256 = (
            new_digest if (slot in bound and new_digest is not None) else raw["registry_sha256"]
        )
        # Validation and payload-digest computation run in EITHER mode --
        # only the write below is gated on dry_run — so a dry run exercises
        # the same required-field checks and payload hashing a real run
        # does, rather than reporting "would migrate" and then failing
        # partway through the real run.
        manifest = _stamp_slot(slot, raw, tier=label, registry_sha256=registry_sha256)
        if not dry_run:
            st = slot.stat()
            write_manifest(slot, manifest)
            os.utime(slot, (st.st_atime, st.st_mtime))

    if dry_run:
        logger.info(
            "[dry-run] %s: would migrate %d slot(s) (%s)%s",
            label,
            len(to_stamp),
            ", ".join(s.name for s in to_stamp),
            "; would rewrite registry" if registry_needs_write else "",
        )
        return _TierResult(label, "would_migrate", tuple(s.name for s in to_stamp))

    if registry_needs_write:
        registry_obj.save_from_bytes(new_registry_bytes, registry_path)

    return _TierResult(label, "migrated", tuple(s.name for s in to_stamp))


def _migrate_registry_only(label: str, registry_path: Path, *, dry_run: bool) -> _TierResult:
    """Re-serialize a tier's registry with ZERO slot candidates present.

    The pre-upgrade simulate-venue shape: content, no manifest, by
    construction. Nothing is invented for it — no manifest, no slot — the
    ruling is that it is not carried forward; boot leaves it unpublishable
    loudly on its own. Only writes when the canonical re-serialization
    actually changes the bytes.
    """
    try:
        raw_bytes = read_maybe_encrypted(registry_path)
    except (RuntimeError, pyrage.DecryptError, OSError) as exc:
        raise TierMigrationBlocked(
            label, f"cannot read registry at {registry_path}: {exc}"
        ) from exc
    registry_obj, new_bytes = _migrate_registry_bytes(
        raw_bytes, tier=label, registry_path=registry_path
    )
    if new_bytes == raw_bytes:
        return _TierResult(label, "already_migrated_registry_only")
    if dry_run:
        logger.info(
            "[dry-run] %s: would rewrite registry-only tier (no slot manifest present, "
            "not carried forward)",
            label,
        )
        return _TierResult(label, "would_migrate_registry_only")
    registry_obj.save_from_bytes(new_bytes, registry_path)
    return _TierResult(label, "migrated_registry_only")


def migrate(adapter_root: Path, *, dry_run: bool = False) -> list[_TierResult]:
    """Run the migration over every tier root and donor store under *adapter_root*.

    Args:
        adapter_root: Adapter store root (production: ``config.adapter_dir``).
        dry_run: Read-only — report without writing.

    Returns:
        One :class:`_TierResult` per tier/store walked.

    Raises:
        TierMigrationBlocked: See :func:`_migrate_tier` — the first
            occurrence stops the whole run; no partial tolerance.
    """
    results: list[_TierResult] = []
    for tier_name, tier_root in iter_tier_roots(adapter_root):
        results.append(_migrate_tier(tier_name, tier_root, dry_run=dry_run))
    for store_name, store_dir in iter_donor_stores(adapter_root):
        results.append(_migrate_tier(store_name, store_dir, dry_run=dry_run))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("data/ha/adapters"),
        help="Adapter store root holding the tier/donor stores (default: data/ha/adapters)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching the filesystem",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.dry_run:
        alive = _pgrep_alive(_LIVE_PROCESS_PGREP_PATTERNS)
        server_up = _server_active()
        if alive or server_up:
            culprits = (["paramem-server (active)"] if server_up else []) + [
                f"PID {pid} ({pat})" for pid, pat in alive
            ]
            logger.error(
                "ERROR: %s alive. Stop the server first "
                "(systemctl --user stop paramem-server) and pause training "
                "(tpause) before rerunning.",
                ", ".join(culprits),
            )
            return 1

    if not args.adapter_root.exists():
        logger.error("adapter_root %s does not exist", args.adapter_root)
        return 1

    try:
        results = migrate(args.adapter_root, dry_run=args.dry_run)
    except TierMigrationBlocked as exc:
        logger.error("STOPPED at tier %r: %s", exc.tier, exc.reason)
        return 1

    by_action: dict[str, int] = {}
    for r in results:
        by_action[r.action] = by_action.get(r.action, 0) + 1
        if r.migrated_slots:
            logger.info("%s: %s (%s)", r.label, r.action, ", ".join(r.migrated_slots))
    logger.info(
        "Done: %s", ", ".join(f"{count} {action}" for action, count in sorted(by_action.items()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
