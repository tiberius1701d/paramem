"""Query routing — selects adapters and keys based on query entities.

Routes queries to the right adapter(s) and identifies which indexed keys
to probe, using the knowledge graph's entity-to-key mappings. This avoids
probing all keys for every query at scale.

Three routing strategies:
  - direct: no entities found, answer from adapter weights (no probing)
  - targeted_probe: entities found, probe specific keys per adapter
  - temporal: time reference detected, probe keys by date range

The router is stateless per query — all state lives in the indexes built
from keyed_pairs.json and the knowledge graph at startup/reload.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Minimum fuzzy match score (0-100) for entity extraction fallback
FUZZY_THRESHOLD = 80
# Maximum keys to probe per query (bounds latency)
MAX_KEYS_PER_QUERY = 10


@dataclass
class RoutingStep:
    """One step in a routing plan: activate an adapter and probe keys."""

    adapter_name: str
    keys_to_probe: list[str] = field(default_factory=list)


@dataclass
class RoutingPlan:
    """What to activate and probe for a given query."""

    steps: list[RoutingStep] = field(default_factory=list)
    strategy: str = "direct"
    matched_entities: list[str] = field(default_factory=list)


class QueryRouter:
    """Routes queries to adapters and keys using the knowledge graph.

    Built from persisted keyed_pairs.json files (one per adapter) and
    the cumulative knowledge graph. Rebuilt after each consolidation.
    """

    def __init__(
        self,
        adapter_dir: Path,
        graph_path: Path | None = None,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.graph_path = Path(graph_path) if graph_path else None

        # adapter_name -> {entity_lower -> set[key]}
        self._entity_key_index: dict[str, dict[str, set[str]]] = {}
        # All known entity names (lowercase) for fast matching
        self._all_entities: set[str] = set()
        # Ordered list for fuzzy matching
        self._entity_list: list[str] = []

        self.reload()

    def reload(self) -> None:
        """Rebuild indexes from disk. Call after consolidation."""
        self._entity_key_index.clear()
        self._all_entities.clear()

        # Scan adapter directories for keyed_pairs.json
        if not self.adapter_dir.exists():
            logger.info("No adapter directory at %s", self.adapter_dir)
            return

        for kp_path in self.adapter_dir.rglob("keyed_pairs.json"):
            # Infer adapter name from directory structure
            # Expected: adapter_dir/adapter_name/keyed_pairs.json
            # or: adapter_dir/keyed_pairs.json (single adapter)
            rel = kp_path.relative_to(self.adapter_dir)
            if len(rel.parts) == 1:
                adapter_name = "episodic"
            else:
                adapter_name = rel.parts[0]

            self._load_keyed_pairs(adapter_name, kp_path)

        # Also load from graph nodes if available
        if self.graph_path and self.graph_path.exists():
            self._load_graph_entities()

        self._entity_list = sorted(self._all_entities)
        logger.info(
            "Router loaded: %d entities, %d adapters indexed",
            len(self._all_entities),
            len(self._entity_key_index),
        )

    def _load_keyed_pairs(self, adapter_name: str, path: Path) -> None:
        """Index entities from a keyed_pairs.json file."""
        try:
            with open(path) as f:
                pairs = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", path, e)
            return

        index = self._entity_key_index.setdefault(adapter_name, {})
        for kp in pairs:
            key = kp.get("key", "")
            for field_name in ("source_subject", "source_object"):
                entity = kp.get(field_name, "")
                if entity and len(entity) > 1:
                    entity_lower = entity.lower().strip()
                    index.setdefault(entity_lower, set()).add(key)
                    self._all_entities.add(entity_lower)

        logger.info(
            "Indexed %s: %d keys, %d entities",
            adapter_name,
            len(pairs),
            len(index),
        )

    def _load_graph_entities(self) -> None:
        """Add graph node names to the entity set for matching."""
        try:
            import networkx as nx

            with open(self.graph_path) as f:
                data = json.load(f)
            graph = nx.node_link_graph(data)
            for node in graph.nodes:
                if isinstance(node, str) and len(node) > 1:
                    self._all_entities.add(node.lower().strip())
        except Exception as e:
            logger.warning("Failed to load graph entities: %s", e)

    def route(self, text: str) -> RoutingPlan:
        """Route a query to the appropriate adapter(s) and keys.

        Returns a RoutingPlan describing what to activate and probe.
        """
        if not self._entity_key_index:
            return RoutingPlan(strategy="direct")

        entities = self._extract_entities(text)
        if not entities:
            return RoutingPlan(strategy="direct")

        adapter_keys = self._resolve_keys(entities)
        if not adapter_keys:
            return RoutingPlan(
                strategy="direct",
                matched_entities=entities,
            )

        # Build steps: semantic first (consolidated), then episodic (recent)
        steps = []
        for adapter_name in ["semantic", "episodic"]:
            if adapter_name in adapter_keys:
                keys = list(adapter_keys[adapter_name])[:MAX_KEYS_PER_QUERY]
                steps.append(
                    RoutingStep(
                        adapter_name=adapter_name,
                        keys_to_probe=keys,
                    )
                )

        # Include any other adapters (personas, etc.)
        for adapter_name, keys in adapter_keys.items():
            if adapter_name not in ("semantic", "episodic"):
                steps.append(
                    RoutingStep(
                        adapter_name=adapter_name,
                        keys_to_probe=list(keys)[:MAX_KEYS_PER_QUERY],
                    )
                )

        return RoutingPlan(
            steps=steps,
            strategy="targeted_probe",
            matched_entities=entities,
        )

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity names from query text.

        First tries exact substring matching against known entity names.
        Falls back to fuzzy matching for near-misses.
        """
        text_lower = text.lower()
        matched = set()

        # Exact substring match (fast path)
        for entity in self._entity_list:
            if entity in text_lower:
                matched.add(entity)

        if matched:
            return sorted(matched)

        # Fuzzy match on individual words and bigrams (fallback)
        words = text_lower.split()
        candidates = words + [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]

        for candidate in candidates:
            for entity in self._entity_list:
                score = fuzz.ratio(candidate, entity)
                if score >= FUZZY_THRESHOLD:
                    matched.add(entity)

        return sorted(matched)

    def _resolve_keys(self, entities: list[str]) -> dict[str, set[str]]:
        """Map entities to keys, grouped by adapter."""
        result: dict[str, set[str]] = {}

        for entity in entities:
            for adapter_name, index in self._entity_key_index.items():
                keys = index.get(entity, set())
                if keys:
                    result.setdefault(adapter_name, set()).update(keys)

        return result

    @property
    def entity_count(self) -> int:
        return len(self._all_entities)

    @property
    def adapter_count(self) -> int:
        return len(self._entity_key_index)
