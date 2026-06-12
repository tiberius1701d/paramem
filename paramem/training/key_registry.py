"""Per-tier key registry for key-addressable replay.

Each adapter tier (episodic, semantic, procedural, episodic_interim_<stamp>)
gets its own :class:`KeyRegistry` instance.  The owning loop holds these in
a ``dict[tier_name, KeyRegistry]`` and persists them as
``<adapter_dir>/<tier>/indexed_key_registry.json`` — tier ownership is
encoded by the file path, not by a field on the record.

The registry's contents are scoped to one tier:

- ``active_keys`` — keys assigned to this tier.
- ``fidelity_history`` — per-key reconstruction-fidelity scores.
- ``health`` — this tier's learning-capacity status (single record).

Cross-tier operations (which tier owns key X, the full health map across
tiers, dropping interim-tier registries at the end of a full cycle) live
on the :class:`paramem.training.consolidation.ConsolidationLoop` since the
loop is what holds the per-tier dict.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Adapter-health status vocabulary. ``"healthy"`` is the default after a
# successful training pass; ``"degenerated"`` is set when the recall
# sanity check trips and the adapter must stop absorbing new keys until
# the next full consolidation cycle.
ADAPTER_HEALTH_HEALTHY = "healthy"
ADAPTER_HEALTH_DEGENERATED = "degenerated"


class KeyRegistry:
    """Tracks one tier's active keys, per-key fidelity, and tier health.

    Keys can be in one of three states:
    - **active**: in ``_active_keys``, enumerated by all normal paths.
    - **stale**: in ``_stale``, excluded from enumeration and the SimHash gate,
      retained for the stale-echo research seam and id-recycling via
      ``get_reclaimable``.  Named :meth:`stale` (not ``mark_stale``) to avoid
      colliding with the dead free function ``persistence.mark_stale``.
    - **removed**: not present anywhere; via :meth:`remove` (hard erasure,
      used by ``/forget`` and reclaim).
    """

    def __init__(self) -> None:
        self._active_keys: list[str] = []
        self._fidelity_history: dict[str, list[float]] = defaultdict(list)
        # Single health record for this tier (None = untracked = healthy).
        self._health: dict | None = None
        # Stale partition: key -> {"stale_since": ISO, "stale_cycles": int}.
        # Keys here are EXCLUDED from normal enumeration and the SimHash gate.
        # Their simhash entries are retained on disk for the stale-echo seam.
        self._stale: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Active-key set
    # ------------------------------------------------------------------

    def add(self, key: str) -> None:
        """Register a new active key for this tier (idempotent)."""
        if key not in self._active_keys:
            self._active_keys.append(key)

    def remove(self, key: str) -> None:
        """Hard-remove a key from this tier (active list, stale set, fidelity).

        Used by ``/forget`` (privacy erasure) and the reclaim path.  A removed
        key is GONE — neither active nor stale.  Does not raise on absent keys.
        """
        self._active_keys = [k for k in self._active_keys if k != key]
        self._fidelity_history.pop(key, None)
        self._stale.pop(key, None)

    def stale(self, key: str) -> None:
        """Move *key* from active to the stale partition (idempotent).

        A stale key is excluded from :meth:`list_active`, :meth:`__contains__`,
        and :meth:`__len__`, but retained in ``_stale`` for the stale-echo probe
        seam.  Its simhash entry is kept on disk.

        :meth:`stale_cycles` starts at 0.  Calling ``stale`` on an already-stale
        or absent key is a no-op.
        """
        if key in self._active_keys:
            self._active_keys = [k for k in self._active_keys if k != key]
            self._fidelity_history.pop(key, None)
            if key not in self._stale:
                self._stale[key] = {
                    "stale_since": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "stale_cycles": 0,
                }

    def knows(self, key: str) -> bool:
        """True when *key* is legitimately tracked by this tier — active OR stale.

        Distinct from :meth:`__contains__` (active-only, serving semantics): a
        stale key is still KNOWN — its simhash and key_metadata are retained on
        disk for the stale-echo seam.  Membership-legitimacy consumers (orphan
        checks, bookkeeping retention) must use this; serving/enumeration
        consumers keep using :meth:`__contains__` / :meth:`list_active`.
        """
        return key in self._active_keys or key in self._stale

    def list_known(self) -> list[str]:
        """All keys this tier legitimately tracks — active first, then stale.

        Each group is in registration/insertion order.  Equivalent to
        :meth:`list_active` + :meth:`list_stale` but expressed as a single call
        so callers can canonically enumerate active ∪ stale without hand-rolling
        the union.
        """
        return list(self._active_keys) + list(self._stale.keys())

    def list_active(self) -> list[str]:
        """Return all active keys in this tier in registration order."""
        return list(self._active_keys)

    def list_stale(self) -> list[str]:
        """Return all stale key ids in this tier."""
        return list(self._stale.keys())

    def is_stale(self, key: str) -> bool:
        """Return True when *key* is in the stale partition of this tier."""
        return key in self._stale

    def get_reclaimable(self, min_stale_cycles: int) -> list[str]:
        """Return stale keys whose ``stale_cycles`` >= *min_stale_cycles*.

        # STALE-RECLAIM SEAM — the reclaim *tick* that actually recycles
        # stale key-ids is deliberately deferred.  This primitive is wired
        # so the on-disk ``stale_cycles`` field advances truthfully (via
        # ``increment_stale_cycles``); the reclaim caller is not yet built.
        """
        return [
            k for k, rec in self._stale.items() if rec.get("stale_cycles", 0) >= min_stale_cycles
        ]

    def increment_stale_cycles(self) -> None:
        """Advance ``stale_cycles`` by 1 for every entry in the stale partition.

        # STALE-RECLAIM SEAM — called by the fold finalize after a durable
        # write so un-persisted stale sets do not advance decay on abort.
        Called at fold N: a key staled in fold N has stale_cycles=0 at the
        durable write; stale_cycles=1 after this call (unobservable until
        fold N+1 reads it from disk).
        """
        for rec in self._stale.values():
            rec["stale_cycles"] = rec.get("stale_cycles", 0) + 1

    def __len__(self) -> int:
        return len(self._active_keys)

    def __contains__(self, key: str) -> bool:
        return key in self._active_keys

    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------

    def update_fidelity(self, key: str, score: float) -> None:
        """Record a reconstruction-fidelity sample for ``key``."""
        self._fidelity_history[key].append(score)

    def get_fidelity_history(self, key: str) -> list[float]:
        """Return the full fidelity series for ``key`` (copy)."""
        return list(self._fidelity_history.get(key, []))

    def get_latest_fidelity(self, key: str) -> float | None:
        """Return the most recent fidelity score for ``key``, or ``None``."""
        history = self._fidelity_history.get(key, [])
        return history[-1] if history else None

    def should_retire(
        self,
        key: str,
        threshold: float = 0.1,
        consecutive_cycles: int = 3,
    ) -> bool:
        """``True`` when the last ``consecutive_cycles`` scores are below threshold."""
        history = self._fidelity_history.get(key, [])
        if len(history) < consecutive_cycles:
            return False
        return all(score < threshold for score in history[-consecutive_cycles:])

    # ------------------------------------------------------------------
    # Tier health (single record; the tier is encoded by the file path)
    # ------------------------------------------------------------------

    def set_health(
        self,
        status: str,
        *,
        reason: str = "",
        keys_at_mark: int | None = None,
    ) -> None:
        """Record the learning-capacity status of this tier.

        Args:
            status: One of :data:`ADAPTER_HEALTH_HEALTHY` /
                :data:`ADAPTER_HEALTH_DEGENERATED`.
            reason: Human-readable justification surfaced in pstatus.
            keys_at_mark: Key count at the time the status was set.
                Defaults to the current count.
        """
        if keys_at_mark is None:
            keys_at_mark = len(self._active_keys)
        self._health = {
            "status": status,
            "reason": reason,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "keys_at_mark": int(keys_at_mark),
        }

    def get_health(self) -> dict | None:
        """Return a copy of this tier's health record, or ``None`` if untracked."""
        return dict(self._health) if self._health is not None else None

    def is_healthy(self) -> bool:
        """``True`` unless the tier is explicitly marked degenerated.

        Untracked (no record yet) is considered healthy — a freshly minted
        registry has no record until someone calls :meth:`set_health`.
        """
        if self._health is None:
            return True
        return self._health.get("status") != ADAPTER_HEALTH_DEGENERATED

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist this tier's registry to ``path`` (encryption-aware)."""
        from paramem.backup.encryption import write_infra_bytes

        payload = self.save_bytes()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_infra_bytes(path, payload)
        logger.info("Key registry saved to %s (%d keys)", path, len(self._active_keys))

    def save_bytes(self) -> bytes:
        """Serialize this tier's registry to canonical UTF-8 JSON bytes.

        The bytes are what :meth:`save` writes.  The atomic consolidation path
        hashes these bytes to obtain ``registry_sha256`` before writing the
        adapter manifest, then calls :meth:`save_from_bytes` with the same
        payload so the on-disk hash is byte-identical to the manifested one.

        The ``"stale"`` field is additive: old files that lack it load as
        ``{}`` via :meth:`load`'s ``data.get("stale", {})``.  Old readers
        simply ignore the extra key (downgrade unsupported, acceptable).
        """
        data = {
            "active_keys": self._active_keys,
            "fidelity_history": dict(self._fidelity_history),
            "health": dict(self._health) if self._health is not None else None,
            "stale": dict(self._stale),
        }
        return json.dumps(data, indent=2).encode("utf-8")

    def save_from_bytes(
        self,
        payload: bytes,
        path: str | Path,
        *,
        _require_consolidating: bool = True,
        consolidating: bool = True,
    ) -> None:
        """Write pre-serialized registry bytes to ``path``.

        Second half of the serialization-barrier split: the bytes must
        come from :meth:`save_bytes` so the on-disk content is byte-identical
        to whatever was hashed for the manifest.

        Raises:
            RuntimeError: When ``_require_consolidating=True`` and
                ``consolidating=False`` — prevents accidental registry writes
                outside the serialization-guarded consolidation window.
        """
        if _require_consolidating and not consolidating:
            raise RuntimeError(
                "KeyRegistry.save_from_bytes called outside a consolidation window "
                "(_require_consolidating=True but consolidating=False)."
            )
        from paramem.backup.encryption import write_infra_bytes

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_infra_bytes(path, payload)
        logger.info("Key registry written from bytes to %s (%d bytes)", path, len(payload))

    @classmethod
    def load(cls, path: str | Path) -> "KeyRegistry":
        """Load a tier's registry from ``path`` (empty registry if absent)."""
        from paramem.backup.encryption import read_maybe_encrypted

        path = Path(path)
        registry = cls()
        if not path.exists():
            logger.info("No registry at %s, starting fresh", path)
            return registry

        data = json.loads(read_maybe_encrypted(path).decode("utf-8"))

        registry._active_keys = data.get("active_keys", [])
        for key, scores in data.get("fidelity_history", {}).items():
            registry._fidelity_history[key] = scores
        health = data.get("health")
        if isinstance(health, dict) and "status" in health:
            registry._health = dict(health)
        # Backward-compat: pre-existing files have no "stale" key → load as {}.
        stale_raw = data.get("stale", {})
        if isinstance(stale_raw, dict):
            registry._stale = {k: dict(v) for k, v in stale_raw.items() if isinstance(v, dict)}

        logger.info(
            "Key registry loaded from %s: %d active keys, %d stale, health=%s",
            path,
            len(registry._active_keys),
            len(registry._stale),
            "set" if registry._health is not None else "unset",
        )
        return registry
