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
    """

    def __init__(self):
        self._active_keys: list[str] = []
        # key -> list of fidelity scores (one per cycle)
        self._fidelity_history: dict[str, list[float]] = defaultdict(list)

    def add(self, key: str) -> None:
        """Register a new active key."""
        if key not in self._active_keys:
            self._active_keys.append(key)

    def remove(self, key: str) -> None:
        """Remove a key from the active set."""
        self._active_keys = [k for k in self._active_keys if k != key]
        self._fidelity_history.pop(key, None)

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

    def save(self, path: str | Path) -> None:
        """Persist registry to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_keys": self._active_keys,
            "fidelity_history": dict(self._fidelity_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Key registry saved to %s (%d keys)", path, len(self._active_keys))

    @classmethod
    def load(cls, path: str | Path) -> "KeyRegistry":
        """Load registry from JSON. Returns empty registry if file missing."""
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

        logger.info(
            "Key registry loaded from %s: %d active keys",
            path,
            len(registry._active_keys),
        )
        return registry
