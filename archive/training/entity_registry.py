"""Entity registry for entity-keyed replay.

Tracks active entities and per-entity reconstruction fidelity
across consolidation cycles. Entities are the natural unit of
memory — knowledge accumulates around people, projects, and
organizations, not sessions.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EntityEntry:
    """Metadata for a tracked entity."""

    name: str
    first_seen: str = ""
    last_seen: str = ""
    session_count: int = 0
    tier: str = "episodic"
    relation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "session_count": self.session_count,
            "tier": self.tier,
            "relation_count": self.relation_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityEntry":
        return cls(
            name=data["name"],
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
            session_count=data.get("session_count", 0),
            tier=data.get("tier", "episodic"),
            relation_count=data.get("relation_count", 0),
        )


class EntityRegistry:
    """Tracks active entities and their reconstruction fidelity.

    Each entity represents a person, project, organization, or other
    knowledge unit whose profile is stored in the adapter weights.
    The registry is lightweight metadata — the knowledge itself lives
    in the adapter.
    """

    def __init__(self):
        self._entities: dict[str, EntityEntry] = {}
        self._fidelity_history: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, session_id: str, relation_count: int = 0) -> None:
        """Register a new entity or update an existing one."""
        if name not in self._entities:
            self._entities[name] = EntityEntry(
                name=name,
                first_seen=session_id,
                last_seen=session_id,
                session_count=1,
                relation_count=relation_count,
            )
        else:
            self.update(name, session_id, relation_count)

    def update(self, name: str, session_id: str, relation_count: int = 0) -> None:
        """Update an existing entity with new session data."""
        if name not in self._entities:
            self.add(name, session_id, relation_count)
            return
        entry = self._entities[name]
        entry.last_seen = session_id
        entry.session_count += 1
        if relation_count > 0:
            entry.relation_count = relation_count

    def remove(self, name: str) -> None:
        """Remove an entity from the registry."""
        self._entities.pop(name, None)
        self._fidelity_history.pop(name, None)

    def list_active(self) -> list[str]:
        """Return all active entity names."""
        return list(self._entities.keys())

    def list_by_tier(self, tier: str) -> list[str]:
        """Return entity names filtered by tier (episodic/semantic)."""
        return [name for name, entry in self._entities.items() if entry.tier == tier]

    def get(self, name: str) -> EntityEntry | None:
        """Get entity metadata, or None if not registered."""
        return self._entities.get(name)

    def set_tier(self, name: str, tier: str) -> None:
        """Update an entity's tier (episodic/semantic)."""
        if name in self._entities:
            self._entities[name].tier = tier

    def update_fidelity(self, name: str, score: float) -> None:
        """Record reconstruction fidelity after a cycle."""
        self._fidelity_history[name].append(score)

    def get_fidelity_history(self, name: str) -> list[float]:
        """Return the fidelity score history for an entity."""
        return list(self._fidelity_history.get(name, []))

    def get_latest_fidelity(self, name: str) -> float | None:
        """Return the most recent fidelity score, or None."""
        history = self._fidelity_history.get(name, [])
        return history[-1] if history else None

    def should_retire(
        self,
        name: str,
        threshold: float = 0.1,
        consecutive_cycles: int = 5,
    ) -> bool:
        """Check if an entity should be retired due to sustained low fidelity."""
        history = self._fidelity_history.get(name, [])
        if len(history) < consecutive_cycles:
            return False
        recent = history[-consecutive_cycles:]
        return all(score < threshold for score in recent)

    def __len__(self) -> int:
        return len(self._entities)

    def __contains__(self, name: str) -> bool:
        return name in self._entities

    def save(self, path: str | Path) -> None:
        """Persist registry to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": {name: entry.to_dict() for name, entry in self._entities.items()},
            "fidelity_history": dict(self._fidelity_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Entity registry saved to %s (%d entities)", path, len(self._entities))

    @classmethod
    def load(cls, path: str | Path) -> "EntityRegistry":
        """Load registry from JSON. Returns empty registry if file missing."""
        path = Path(path)
        registry = cls()
        if not path.exists():
            logger.info("No entity registry at %s, starting fresh", path)
            return registry

        with open(path) as f:
            data = json.load(f)

        for name, entry_data in data.get("entities", {}).items():
            registry._entities[name] = EntityEntry.from_dict(entry_data)
        for name, scores in data.get("fidelity_history", {}).items():
            registry._fidelity_history[name] = scores

        logger.info(
            "Entity registry loaded from %s: %d entities",
            path,
            len(registry._entities),
        )
        return registry
