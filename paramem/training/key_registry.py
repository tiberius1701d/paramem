"""Key registry for key-addressable replay.

Manages the set of active memory keys and tracks per-key
reconstruction fidelity across consolidation cycles.
"""

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
    """Tracks active memory keys and their reconstruction fidelity.

    Each key maps to a session's knowledge graph stored in the adapter.
    The registry is the only external metadata — the knowledge itself
    lives in the adapter weights.

    Each key also records which adapter owns it via ``adapter_id``.
    The value is a string such as ``"main"`` (for consolidated main
    adapters) or ``"episodic_interim_20260418T0900"`` (for a rolling
    interim adapter).  Legacy registries loaded from disk that lack this
    field default every key to ``"main"`` so existing deployments are
    unaffected.

    The registry also carries an ``adapter_health`` map keyed by
    ``adapter_id`` that records the learning-capacity status of each
    adapter.  This is intentionally colocated with the key data so it
    benefits from the same atomic-persist discipline (registry is the
    "commit" signal — see consolidation.py's I5 ordering).  No personal
    data leaks into health metadata, so storing it in the registry does
    not widen the privacy surface.
    """

    def __init__(self):
        self._active_keys: list[str] = []
        # key -> list of fidelity scores (one per cycle)
        self._fidelity_history: dict[str, list[float]] = defaultdict(list)
        # key -> adapter_id string ("main" by default)
        self._adapter_id: dict[str, str] = {}
        # adapter_id -> {"status", "reason", "updated_at", "keys_at_mark"}
        self._adapter_health: dict[str, dict] = {}

    def add(self, key: str, adapter_id: str = "main") -> None:
        """Register a new active key.

        Args:
            key: The indexed key string (e.g. ``"graph1"``).
            adapter_id: Which adapter owns this key.  Defaults to
                ``"main"`` so existing positional callers are unaffected.
        """
        if key not in self._active_keys:
            self._active_keys.append(key)
        # Always update adapter_id (handles re-assignment on consolidation).
        self._adapter_id[key] = adapter_id

    def remove(self, key: str) -> None:
        """Remove a key from the active set, clearing all associated metadata."""
        self._active_keys = [k for k in self._active_keys if k != key]
        self._fidelity_history.pop(key, None)
        self._adapter_id.pop(key, None)

    def list_active(self) -> list[str]:
        """Return all active keys in registration order."""
        return list(self._active_keys)

    def __len__(self) -> int:
        return len(self._active_keys)

    def __contains__(self, key: str) -> bool:
        return key in self._active_keys

    def update_fidelity(self, key: str, score: float) -> None:
        """Record reconstruction fidelity for a key after a cycle."""
        self._fidelity_history[key].append(score)

    def get_fidelity_history(self, key: str) -> list[float]:
        """Return the fidelity score history for a key."""
        return list(self._fidelity_history.get(key, []))

    def get_latest_fidelity(self, key: str) -> float | None:
        """Return the most recent fidelity score, or None if no history."""
        history = self._fidelity_history.get(key, [])
        return history[-1] if history else None

    def should_retire(
        self,
        key: str,
        threshold: float = 0.1,
        consecutive_cycles: int = 3,
    ) -> bool:
        """Check if a key should be retired due to sustained low fidelity.

        Returns True if the last `consecutive_cycles` fidelity scores
        are all below `threshold`.
        """
        history = self._fidelity_history.get(key, [])
        if len(history) < consecutive_cycles:
            return False
        recent = history[-consecutive_cycles:]
        return all(score < threshold for score in recent)

    def set_adapter_id(self, key: str, adapter_id: str) -> None:
        """Overwrite the adapter_id for an existing key.

        Used during consolidation to reassign a key from a session adapter
        to its new main adapter after a weekly refresh.

        Args:
            key: The indexed key to update.
            adapter_id: The new adapter owner (e.g. ``"episodic"``).
        """
        self._adapter_id[key] = adapter_id

    def get_adapter_id(self, key: str) -> str:
        """Return the adapter_id for a key, defaulting to ``"main"``.

        The ``"main"`` default ensures backward compatibility with keys
        registered before this field was introduced.

        Args:
            key: The indexed key to look up.

        Returns:
            The adapter_id string, or ``"main"`` if not recorded.
        """
        return self._adapter_id.get(key, "main")

    def keys_for_adapter(self, adapter_id: str) -> list[str]:
        """Return all active keys owned by the given adapter.

        Args:
            adapter_id: The adapter to filter by (e.g. ``"episodic"`` or
                ``"episodic_interim_20260418T0900"``).

        Returns:
            Keys in registration order that belong to ``adapter_id``.
        """
        return [k for k in self._active_keys if self._adapter_id.get(k, "main") == adapter_id]

    # ------------------------------------------------------------------
    # Adapter health
    # ------------------------------------------------------------------

    def set_adapter_health(
        self,
        adapter_id: str,
        status: str,
        *,
        reason: str = "",
        keys_at_mark: int | None = None,
    ) -> None:
        """Record the current learning-capacity status of an adapter.

        Args:
            adapter_id: Adapter name (``"episodic_interim_<stamp>"`` or a
                main-tier name).
            status: One of :data:`ADAPTER_HEALTH_HEALTHY` or
                :data:`ADAPTER_HEALTH_DEGENERATED`.
            reason: Human-readable justification (e.g. ``"recall 0.78 <
                0.95 after retrain on 42 keys"``).  Surfaced in pstatus.
            keys_at_mark: Number of keys the adapter was holding when the
                status was set — makes the capacity ceiling observable
                across runs.  Defaults to current key count for the adapter.
        """
        if keys_at_mark is None:
            keys_at_mark = sum(1 for v in self._adapter_id.values() if v == adapter_id)
        self._adapter_health[adapter_id] = {
            "status": status,
            "reason": reason,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "keys_at_mark": int(keys_at_mark),
        }

    def get_adapter_health(self, adapter_id: str) -> dict | None:
        """Return the health record for *adapter_id* or ``None`` if untracked.

        An untracked adapter is treated as healthy by :meth:`is_adapter_healthy`.
        """
        record = self._adapter_health.get(adapter_id)
        return dict(record) if record is not None else None

    def list_adapter_health(self) -> dict[str, dict]:
        """Return a copy of the full adapter_id → health-record map.

        Used by ``/status`` / pstatus to surface every known health entry
        (healthy + degenerated) without exposing the mutable internal
        dict.
        """
        return {k: dict(v) for k, v in self._adapter_health.items()}

    def is_adapter_healthy(self, adapter_id: str) -> bool:
        """Return True unless *adapter_id* is explicitly marked degenerated.

        Untracked adapters (no record) are considered healthy — a newly
        minted interim adapter starts life absent from the map and
        accrues a record only once someone calls :meth:`set_adapter_health`.
        """
        record = self._adapter_health.get(adapter_id)
        if record is None:
            return True
        return record.get("status") != ADAPTER_HEALTH_DEGENERATED

    def clear_interim_adapter_health(self) -> int:
        """Drop every health entry whose ``adapter_id`` is an interim adapter.

        Called at the end of a full consolidation cycle: every
        ``*_interim_*`` adapter has just been collapsed into its main
        tier, so any lingering health records would describe adapters
        that no longer exist.  Main-tier records are preserved — they
        still matter across cycles.

        Returns:
            Number of entries removed.
        """
        interim_keys = [k for k in self._adapter_health if "_interim_" in k]
        for k in interim_keys:
            del self._adapter_health[k]
        return len(interim_keys)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist registry to JSON.

        Writes ``active_keys``, ``fidelity_history``, ``adapter_id``, and
        ``adapter_health`` so every piece of per-key / per-adapter
        metadata survives a process restart.
        """
        payload = self.save_bytes()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        logger.info("Key registry saved to %s (%d keys)", path, len(self._active_keys))

    def save_bytes(self) -> bytes:
        """Serialize the registry to UTF-8 JSON bytes WITHOUT writing to disk.

        Returns the canonical bytes that :meth:`save` would write.  Callers
        in the I5 consolidation path hash these bytes to obtain
        ``registry_sha256`` before writing the adapter manifest, then call
        :meth:`save_from_bytes` with the same payload so the on-disk hash
        is byte-identical to the manifested hash.

        Returns:
            UTF-8 encoded JSON bytes (canonical, sorted keys, indent=2).
        """
        data = {
            "active_keys": self._active_keys,
            "fidelity_history": dict(self._fidelity_history),
            "adapter_id": dict(self._adapter_id),
            "adapter_health": {k: dict(v) for k, v in self._adapter_health.items()},
        }
        return json.dumps(data, indent=2).encode("utf-8")

    def save_from_bytes(
        self,
        payload: bytes,
        path: "str | Path",
        *,
        _require_consolidating: bool = True,
        consolidating: bool = True,
    ) -> None:
        """Write pre-serialized registry bytes to *path*.

        This is the second half of the I5 serialization-barrier split.
        The bytes must come from :meth:`save_bytes` so the on-disk content
        is byte-identical to whatever was hashed for the manifest.

        Args:
            payload: Bytes from :meth:`save_bytes` (not re-serialized).
            path: Destination file path.
            _require_consolidating: When ``True`` (default), asserts that
                the caller is running inside a consolidation window by
                checking the *consolidating* argument.  Set to ``False``
                for experiment harnesses that do not have a consolidation
                context.
            consolidating: The caller's consolidation-context flag.  In the
                server path, callers read ``_state["consolidating"]`` and
                forward it here.  Experiment callers pass ``False`` alongside
                ``_require_consolidating=False``.

        Raises:
            RuntimeError: When ``_require_consolidating=True`` and
                ``consolidating=False`` — prevents accidental registry writes
                outside the serialization-guarded consolidation window.
        """
        if _require_consolidating and not consolidating:
            raise RuntimeError(
                "KeyRegistry.save_from_bytes called outside a consolidation window "
                "(_require_consolidating=True but consolidating=False).  "
                "This is a defence-in-depth assertion — see plan §2.5."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        logger.info("Key registry written from bytes to %s (%d bytes)", path, len(payload))

    @classmethod
    def load(cls, path: str | Path) -> "KeyRegistry":
        """Load registry from JSON. Returns empty registry if file missing.

        Legacy files that lack ``adapter_id`` or ``adapter_health`` are
        handled gracefully: every key defaults to ``"main"`` for ownership
        and the health map loads empty (all adapters treated as healthy
        by :meth:`is_adapter_healthy`).
        """
        path = Path(path)
        registry = cls()
        if not path.exists():
            logger.info("No registry at %s, starting fresh", path)
            return registry

        with open(path) as f:
            data = json.load(f)

        registry._active_keys = data.get("active_keys", [])
        for key, scores in data.get("fidelity_history", {}).items():
            registry._fidelity_history[key] = scores
        # Backward compat: missing field → every key defaults to "main".
        adapter_id_map = data.get("adapter_id", {})
        for key in registry._active_keys:
            registry._adapter_id[key] = adapter_id_map.get(key, "main")
        # Health map is optional; absent in pre-health-schema registries.
        for adapter_id, record in data.get("adapter_health", {}).items():
            if isinstance(record, dict) and "status" in record:
                registry._adapter_health[adapter_id] = dict(record)

        logger.info(
            "Key registry loaded from %s: %d active keys, %d health entries",
            path,
            len(registry._active_keys),
            len(registry._adapter_health),
        )
        return registry
