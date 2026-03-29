"""HA entity graph — indexes HA devices, areas, and action verbs for routing.

Built from HA REST API responses (/api/states and /api/services) at server
startup. The router uses this graph alongside the PA knowledge graph to
distinguish personal-knowledge queries from device/action queries.

Matching is pure string ops (substring + fuzzy) — zero LLM inference cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Minimum fuzzy match score for HA entity matching
FUZZY_THRESHOLD = 80

# Service name fragments to skip (not user-facing actions)
_SKIP_SERVICES = {"reload", "update", "refresh", "homeassistant"}


@dataclass
class HAEntityNode:
    """A single HA entity with its metadata."""

    entity_id: str  # "light.living_room"
    friendly_name: str  # "Living Room Light"
    domain: str  # "light"
    area: str  # "Living Room" (may be empty)


@dataclass
class HAMatchResult:
    """Result of matching a query against the HA entity graph."""

    matched_entities: list[str]  # friendly names that matched
    matched_areas: list[str]  # area names that matched
    matched_verbs: list[str]  # action verbs that matched
    domains: list[str]  # domains of matched entities/verbs

    @property
    def has_entity_match(self) -> bool:
        return bool(self.matched_entities) or bool(self.matched_areas)

    @property
    def has_verb_match(self) -> bool:
        return bool(self.matched_verbs)


class HAEntityGraph:
    """In-memory index of HA entities, areas, and action verbs.

    Not a networkx graph — flat indexes optimized for the same
    substring/fuzzy matching pattern the PA router uses.
    """

    def __init__(self):
        # All matchable names (friendly names + areas, lowercase)
        self._entity_names: set[str] = set()
        # Sorted for fuzzy matching
        self._entity_list: list[str] = []
        # friendly_name_lower → node
        self._entity_lookup: dict[str, HAEntityNode] = {}
        # Area names (lowercase)
        self._area_names: set[str] = set()
        # domain → set of action verb phrases
        self._domain_verbs: dict[str, set[str]] = {}
        # All verb phrases (lowercase)
        self._all_verbs: set[str] = set()
        # Sorted for fuzzy matching
        self._verb_list: list[str] = []
        # verb → domains that support it
        self._verb_to_domains: dict[str, set[str]] = {}

    @classmethod
    def build(
        cls,
        states: list[dict],
        services: list[dict] | None = None,
    ) -> HAEntityGraph:
        """Build an HA entity graph from raw API responses.

        Args:
            states: Response from HA /api/states endpoint.
            services: Response from HA /api/services endpoint.
        """
        graph = cls()
        graph._index_states(states)
        if services:
            graph._index_services(services)
        graph._entity_list = sorted(graph._entity_names)
        graph._verb_list = sorted(graph._all_verbs)
        logger.info(
            "HA entity graph: %d entities, %d areas, %d verbs across %d domains",
            len(graph._entity_lookup),
            len(graph._area_names),
            len(graph._all_verbs),
            len(graph._domain_verbs),
        )
        return graph

    def refresh(self, states: list[dict], services: list[dict] | None = None) -> None:
        """Rebuild indexes in place from fresh API data."""
        self._entity_names.clear()
        self._entity_list.clear()
        self._entity_lookup.clear()
        self._area_names.clear()
        self._domain_verbs.clear()
        self._all_verbs.clear()
        self._verb_list.clear()
        self._verb_to_domains.clear()
        self._index_states(states)
        if services:
            self._index_services(services)
        self._entity_list = sorted(self._entity_names)
        self._verb_list = sorted(self._all_verbs)
        logger.info(
            "HA graph refreshed: %d entities, %d areas, %d verbs",
            len(self._entity_lookup),
            len(self._area_names),
            len(self._all_verbs),
        )

    def _index_states(self, states: list[dict]) -> None:
        """Extract entities and areas from HA state objects."""
        for state in states:
            entity_id = state.get("entity_id", "")
            if not entity_id:
                continue

            attrs = state.get("attributes", {})
            friendly = attrs.get("friendly_name", "")
            if not friendly:
                continue

            # Domain from entity_id prefix (e.g., "light.living_room" → "light")
            domain = entity_id.split(".", 1)[0] if "." in entity_id else ""

            # Area from attributes (HA exposes this when entity is area-assigned)
            area = attrs.get("area_name", "") or attrs.get("area_id", "")

            node = HAEntityNode(
                entity_id=entity_id,
                friendly_name=friendly,
                domain=domain,
                area=area,
            )

            friendly_lower = friendly.lower()
            self._entity_names.add(friendly_lower)
            self._entity_lookup[friendly_lower] = node

            if area:
                area_lower = area.lower()
                self._area_names.add(area_lower)
                self._entity_names.add(area_lower)

    def _index_services(self, services: list[dict]) -> None:
        """Extract action verbs per domain from HA services response.

        HA /api/services returns a list of dicts, each with:
            {"domain": "light", "services": {"turn_on": {...}, "turn_off": {...}, ...}}
        """
        for entry in services:
            domain = entry.get("domain", "")
            if not domain or domain in _SKIP_SERVICES:
                continue

            svc_dict = entry.get("services", {})
            verbs = set()
            for svc_name in svc_dict:
                if svc_name.startswith("_") or svc_name in _SKIP_SERVICES:
                    continue
                # Convert service name to natural verb phrase
                # "turn_on" → "turn on", "set_temperature" → "set temperature"
                verb = svc_name.replace("_", " ")
                verbs.add(verb)
                self._verb_to_domains.setdefault(verb, set()).add(domain)

            if verbs:
                self._domain_verbs[domain] = verbs
                self._all_verbs.update(verbs)

    def match(self, text: str) -> HAMatchResult | None:
        """Match query text against HA entities, areas, and action verbs.

        Returns HAMatchResult if any HA entity/area/verb matched, None otherwise.
        Uses the same substring + fuzzy pattern as the PA router.
        """
        text_lower = text.lower()

        # Match entities and areas (substring, then fuzzy fallback)
        matched_entities = []
        matched_areas = []
        entity_domains = set()

        for name in self._entity_list:
            if name in text_lower:
                if name in self._area_names:
                    matched_areas.append(name)
                    # Collect domains of entities in this area
                    for fn, node in self._entity_lookup.items():
                        if node.area and node.area.lower() == name:
                            entity_domains.add(node.domain)
                elif name in self._entity_lookup:
                    matched_entities.append(name)
                    entity_domains.add(self._entity_lookup[name].domain)

        # Fuzzy fallback for entities if no exact match
        if not matched_entities and not matched_areas:
            words = text_lower.split()
            candidates = words + [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
            for candidate in candidates:
                for name in self._entity_list:
                    score = fuzz.ratio(candidate, name)
                    if score >= FUZZY_THRESHOLD:
                        if name in self._area_names:
                            matched_areas.append(name)
                        elif name in self._entity_lookup:
                            matched_entities.append(name)
                            entity_domains.add(self._entity_lookup[name].domain)

        # Match action verbs (substring only — verbs are short, fuzzy is noisy)
        matched_verbs = []
        verb_domains = set()
        for verb in self._verb_list:
            if verb in text_lower:
                matched_verbs.append(verb)
                verb_domains.update(self._verb_to_domains.get(verb, set()))

        if not matched_entities and not matched_areas and not matched_verbs:
            return None

        all_domains = sorted(entity_domains | verb_domains)
        return HAMatchResult(
            matched_entities=sorted(matched_entities),
            matched_areas=sorted(matched_areas),
            matched_verbs=sorted(matched_verbs),
            domains=all_domains,
        )

    @property
    def entity_count(self) -> int:
        return len(self._entity_lookup)

    @property
    def area_count(self) -> int:
        return len(self._area_names)

    @property
    def verb_count(self) -> int:
        return len(self._all_verbs)
