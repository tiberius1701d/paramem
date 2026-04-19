"""Key registry for key-addressable replay.

Manages the set of active memory keys and tracks per-key
reconstruction fidelity across consolidation cycles.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


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
    """

    def __init__(self):
        self._active_keys: list[str] = []
        # key -> list of fidelity scores (one per cycle)
        self._fidelity_history: dict[str, list[float]] = defaultdict(list)
        # key -> adapter_id string ("main" by default)
        self._adapter_id: dict[str, str] = {}

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

    def save(self, path: str | Path) -> None:
        """Persist registry to JSON.

        Writes ``active_keys``, ``fidelity_history``, and ``adapter_id``
        so that all per-key metadata survives a process restart.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_keys": self._active_keys,
            "fidelity_history": dict(self._fidelity_history),
            "adapter_id": dict(self._adapter_id),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Key registry saved to %s (%d keys)", path, len(self._active_keys))

    @classmethod
    def load(cls, path: str | Path) -> "KeyRegistry":
        """Load registry from JSON. Returns empty registry if file missing.

        Legacy files that lack the ``adapter_id`` field are handled
        gracefully: every key defaults to ``"main"`` so existing
        deployments load unchanged.
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

        logger.info(
            "Key registry loaded from %s: %d active keys",
            path,
            len(registry._active_keys),
        )
        return registry
