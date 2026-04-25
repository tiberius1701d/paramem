"""Query routing — selects adapters and keys based on query entities.

Routes queries to the right adapter(s) and identifies which indexed keys
to probe, using the knowledge graph's entity-to-key mappings. This avoids
probing all keys for every query at scale.

Dual-graph routing: queries are matched against both the PA knowledge graph
(personal entities) and the HA entity graph (devices, areas, action verbs).
The match_source field in RoutingPlan tells inference.py which path to take:
  - "pa": personal knowledge → local adapter probe
  - "ha": device/action → HA conversation agent
  - "both": PA + HA overlap → PA first, HA on [ESCALATE]
  - "none": no match → SOTA cloud agent

Imperative fallback: when match_source would be "none" and the query is an
imperative command (not a question), it is promoted to "ha" so the HA
conversation agent can attempt to resolve it. Commands like "play music" or
"set a timer" name no specific entity but are almost certainly device actions.
Interrogatives ("What is X?") stay on the "none" → SOTA path.

The router is stateless per query — all state lives in the indexes built
from keyed_pairs.json, the knowledge graph, and the HA entity graph.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

if TYPE_CHECKING:
    from paramem.server.ha_graph import HAEntityGraph, HAMatchResult

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
    # Dual-graph routing: which graph(s) produced the match
    match_source: str = "none"  # "pa", "ha", "both", "none"
    # Action verb + HA entity detected (strong HA signal)
    imperative: bool = False
    # HA domains of matched entities/verbs
    ha_domains: list[str] = field(default_factory=list)


_INTERIM_PREFIX = "episodic_interim_"
_INTERIM_DATE_RE = re.compile(r"^episodic_interim_(\d{8}T\d{4})$")


def _interim_sort_key(adapter_name: str) -> str | None:
    """Extract the YYYYMMDDTHHMM stamp from an episodic_interim_YYYYMMDDTHHMM name.

    Returns the stamp string (used for descending sort in route() so the
    most-recently-created interim adapter is probed first), or None if
    *adapter_name* does not match the interim-adapter naming pattern.
    Non-interim adapters are not affected by this helper.
    """
    m = _INTERIM_DATE_RE.match(adapter_name)
    return m.group(1) if m else None


_INTERROGATIVE_PREFIXES = frozenset(
    {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "is",
        "are",
        "does",
        "do",
        "can",
        "could",
        "would",
        "will",
        "which",
    }
)


# Declarative openings — pronoun- or possessive-fronted statements ("I'm Alex",
# "We had dinner", "My wife is here"). The imperative fallback below uses this
# to avoid misclassifying personal statements as device commands. Imperative
# English typically opens with a base-form verb ("play", "turn on", "set"),
# never with a subject pronoun or possessive.
_DECLARATIVE_PREFIXES = frozenset(
    {
        "i",
        "i'm",
        "i'd",
        "i've",
        "i'll",
        "we",
        "we're",
        "we'd",
        "we've",
        "we'll",
        "you",
        "you're",
        "you'd",
        "you've",
        "you'll",
        "he",
        "he's",
        "he'd",
        "she",
        "she's",
        "she'd",
        "it",
        "it's",
        "it'd",
        "they",
        "they're",
        "they'd",
        "they've",
        "they'll",
        "this",
        "that",
        "these",
        "those",
        "there",
        "my",
        "our",
        "your",
        "his",
        "her",
        "their",
    }
)


def _is_interrogative(text: str) -> bool:
    """Check if the query is a question (not an imperative command)."""
    first_word = text.strip().split()[0].lower() if text.strip() else ""
    # Handle possessives: "who's" → "who", "what's" → "what"
    if first_word.endswith("'s"):
        first_word = first_word[:-2]
    return first_word in _INTERROGATIVE_PREFIXES


def _is_declarative(text: str) -> bool:
    """First-word heuristic for pronoun/possessive-fronted statements."""
    first_word = text.strip().split()[0].lower() if text.strip() else ""
    return first_word in _DECLARATIVE_PREFIXES


class QueryRouter:
    """Routes queries to adapters and keys using the knowledge graph.

    Built from persisted keyed_pairs.json files (one per adapter) and
    the cumulative knowledge graph. Rebuilt after each consolidation.
    Optionally includes an HA entity graph for dual-graph routing.
    """

    def __init__(
        self,
        adapter_dir: Path,
        graph_path: Path | None = None,
        ha_graph: HAEntityGraph | None = None,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.graph_path = Path(graph_path) if graph_path else None
        self._ha_graph = ha_graph

        # adapter_name -> {entity_lower -> set[key]}
        self._entity_key_index: dict[str, dict[str, set[str]]] = {}
        # speaker_id -> set[key] (for speaker-scoped filtering)
        self._speaker_key_index: dict[str, set[str]] = {}
        # All known entity names (lowercase) for fast matching
        self._all_entities: set[str] = set()
        # Ordered list for fuzzy matching
        self._entity_list: list[str] = []

        self.reload()

    def reload(self) -> None:
        """Rebuild indexes from disk. Call after consolidation."""
        self._entity_key_index.clear()
        self._speaker_key_index.clear()
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
        """Index entities from a keyed_pairs.json file.  Transparently
        decrypts age-wrapped content when the daily identity is loaded."""
        from paramem.backup.encryption import read_maybe_encrypted

        try:
            pairs = json.loads(read_maybe_encrypted(path).decode("utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to load %s: %s", path, e)
            return

        index = self._entity_key_index.setdefault(adapter_name, {})
        for kp in pairs:
            key = kp.get("key", "")
            speaker_id = kp.get("speaker_id")
            if speaker_id:
                self._speaker_key_index.setdefault(speaker_id, set()).add(key)
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
        """Add graph node names to the entity set for matching.  Transparently
        decrypts age-wrapped content when the daily identity is loaded."""
        try:
            import networkx as nx

            from paramem.backup.encryption import read_maybe_encrypted

            data = json.loads(read_maybe_encrypted(self.graph_path).decode("utf-8"))
            graph = nx.node_link_graph(data)
            for node in graph.nodes:
                if isinstance(node, str) and len(node) > 1:
                    self._all_entities.add(node.lower().strip())
        except Exception as e:
            logger.warning("Failed to load graph entities: %s", e)

    def route(
        self,
        text: str,
        speaker: str | None = None,
        speaker_id: str | None = None,
    ) -> RoutingPlan:
        """Route a query to the appropriate adapter(s) and keys.

        If speaker is provided, it is always injected as an implicit
        entity so that self-referential queries ("What is my name?")
        resolve to the speaker's keys.

        speaker_id scopes key access: only keys tagged with this speaker_id
        are returned. If speaker_id is None, no personal keys are served.

        Returns a RoutingPlan describing what to activate and probe.
        """
        # Speaker's allowed key set (empty if anonymous)
        allowed_keys = self._speaker_key_index.get(speaker_id, set()) if speaker_id else set()

        # --- PA graph matching ---
        pa_entities = []
        if self._entity_key_index and allowed_keys:
            pa_entities = self._extract_entities(text)

            # Inject speaker as implicit entity
            if speaker:
                speaker_lower = speaker.lower().strip()
                if speaker_lower not in pa_entities and speaker_lower in self._all_entities:
                    pa_entities.append(speaker_lower)

            # One-hop expansion scoped to speaker's keys
            if speaker:
                connected = self._get_connected_entities(speaker.lower().strip(), allowed_keys)
                for entity in connected:
                    if entity not in pa_entities:
                        pa_entities.append(entity)

            pa_entities.sort()

        has_pa = bool(pa_entities) and bool(self._resolve_keys(pa_entities, allowed_keys))

        # --- HA graph matching ---
        ha_match: HAMatchResult | None = None
        if self._ha_graph is not None:
            ha_match = self._ha_graph.match(text)
        has_ha = ha_match is not None and ha_match.has_entity_match

        # --- Determine match_source ---
        if has_pa and has_ha:
            match_source = "both"
        elif has_pa:
            match_source = "pa"
        elif has_ha:
            match_source = "ha"
        else:
            match_source = "none"

        # --- Imperative detection ---
        imperative = False
        if has_ha and ha_match is not None:
            has_verb = ha_match.has_verb_match
            is_question = _is_interrogative(text)
            imperative = has_verb and not is_question

        # --- Imperative fallback: no graph match but clearly a command ---
        # Imperatives with no entity match (e.g. "play music", "set a timer") are
        # almost certainly device commands. Route them to HA so its conversation
        # agent can resolve the intent — it's better equipped than the routing
        # layer to handle commands that don't name a specific entity.
        # Interrogatives stay on the SOTA path (knowledge/reasoning questions).
        # Pronoun- or possessive-fronted statements ("I'm Alex", "My wife
        # is here") are declarative and stay off the HA path — without this
        # gate they were classified as imperatives and routed to HA, which
        # the sanitizer then blocked, leaving _base_model_answer to invent
        # an implicit question to answer.
        if (
            match_source == "none"
            and self._ha_graph is not None
            and not _is_interrogative(text)
            and not _is_declarative(text)
        ):
            match_source = "ha"
            imperative = True
            logger.info("Imperative fallback: no entity match, routing to HA agent")

        # --- Build PA adapter steps (only if PA matched) ---
        steps = []
        if has_pa:
            adapter_keys = self._resolve_keys(pa_entities, allowed_keys)

            # 1. Main adapters in CLAUDE.md inference assembly order:
            #    procedural (preferences) → episodic (recent) → semantic (consolidated).
            #    Procedural-first is load-bearing: behavioral preferences must surface
            #    before any episodic/semantic context for the PA to feel personalized.
            #    See feedback_router_procedural_first.md.
            _MAIN_ORDER = ["procedural", "episodic", "semantic"]
            for adapter_name in _MAIN_ORDER:
                if adapter_name in adapter_keys:
                    keys = list(adapter_keys[adapter_name])[:MAX_KEYS_PER_QUERY]
                    steps.append(
                        RoutingStep(
                            adapter_name=adapter_name,
                            keys_to_probe=keys,
                        )
                    )

            # 2. Interim adapters (episodic_interim_YYYYMMDDTHHMM) newest-first, then
            #    any other non-main adapters in arbitrary order.  Separating the two
            #    groups ensures the stamp-descending sort only touches interim names.
            known = set(_MAIN_ORDER)
            interims: list[tuple[str, set[str]]] = []
            others: list[tuple[str, set[str]]] = []
            for adapter_name, keys in adapter_keys.items():
                if adapter_name not in known:
                    if _interim_sort_key(adapter_name) is not None:
                        interims.append((adapter_name, keys))
                    else:
                        others.append((adapter_name, keys))

            # Newest stamp first — YYYYMMDDTHHMM sorts lexicographically so
            # descending string order gives the most recently created interim first.
            interims.sort(key=lambda item: _interim_sort_key(item[0]) or "", reverse=True)

            for adapter_name, keys in interims + others:
                steps.append(
                    RoutingStep(
                        adapter_name=adapter_name,
                        keys_to_probe=list(keys)[:MAX_KEYS_PER_QUERY],
                    )
                )

        strategy = "targeted_probe" if steps else "direct"

        ha_domains = ha_match.domains if ha_match else []

        if match_source != "none":
            logger.info(
                "Routed query: source=%s, pa_entities=%s, ha=%s, imperative=%s",
                match_source,
                pa_entities if has_pa else [],
                ha_domains if has_ha else [],
                imperative,
            )

        return RoutingPlan(
            steps=steps,
            strategy=strategy,
            matched_entities=pa_entities,
            match_source=match_source,
            imperative=imperative,
            ha_domains=ha_domains,
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

    def _get_connected_entities(
        self, entity: str, allowed_keys: set[str] | None = None
    ) -> list[str]:
        """Find entities one hop away from the given entity via shared keys.

        If entity "alex" has keys where "sam" is the other endpoint,
        "sam" is returned. Only traverses keys in allowed_keys if provided.
        """
        connected = set()
        for index in self._entity_key_index.values():
            keys = index.get(entity, set())
            if allowed_keys is not None:
                keys = keys & allowed_keys
            for other_entity, other_keys in index.items():
                if other_entity != entity:
                    shared = keys & other_keys
                    if allowed_keys is not None:
                        shared = shared & allowed_keys
                    if shared:
                        connected.add(other_entity)
        return sorted(connected)

    def _resolve_keys(
        self, entities: list[str], allowed_keys: set[str] | None = None
    ) -> dict[str, set[str]]:
        """Map entities to keys, grouped by adapter.

        If allowed_keys is provided, only returns keys in the allowed set.
        """
        result: dict[str, set[str]] = {}

        for entity in entities:
            for adapter_name, index in self._entity_key_index.items():
                keys = index.get(entity, set())
                if allowed_keys is not None:
                    keys = keys & allowed_keys
                if keys:
                    result.setdefault(adapter_name, set()).update(keys)

        return result

    @property
    def entity_count(self) -> int:
        return len(self._all_entities)

    @property
    def adapter_count(self) -> int:
        return len(self._entity_key_index)
