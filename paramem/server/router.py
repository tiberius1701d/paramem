"""Query routing — selects adapters and keys based on query entities.

Routes queries to the right adapter(s) and identifies which indexed keys
to probe, using the knowledge graph's entity-to-key mappings.  This
avoids probing all keys for every query at scale.

Dual-graph matching feeds the intent classifier:

* PA knowledge graph (personal entities) → ``has_pa``
* HA entity graph (devices, areas, action verbs) → ``has_ha``

``classify_intent`` collapses these signals into a single
:class:`Intent` value on the returned :class:`RoutingPlan`.  Downstream
routing in ``inference.py`` dispatches on ``plan.intent`` alone; the
graph-coverage details are still surfaced via ``plan.steps`` (PA probe
targets) and ``plan.ha_domains`` (HA observability).

The router is stateless per query — all state lives in the indexes
built from the ConsolidationLoop cache, the knowledge graph, and the
HA entity graph.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

if TYPE_CHECKING:
    from paramem.server.config import IntentConfig
    from paramem.server.ha_graph import HAEntityGraph, HAMatchResult

logger = logging.getLogger(__name__)

# Minimum fuzzy match score (0-100) for entity extraction fallback
FUZZY_THRESHOLD = 80


class Intent(str, Enum):
    """Single explicit routing axis for queries.

    Populated on :class:`RoutingPlan` by future commits; today this enum
    exists so consumers can start preparing to read it.  Default on
    :class:`RoutingPlan` is :attr:`UNKNOWN` — the field is informational
    until the residual classifier is wired in.

    * ``PERSONAL`` — query references the speaker's life or graph; never
      escalates to cloud.
    * ``COMMAND`` — imperative or device query; routed to the HA agent.
    * ``GENERAL`` — general knowledge / real-time data; full
      PA→HA→SOTA→base-model escalation chain available.
    * ``UNKNOWN`` — classifier unavailable or ambiguous; treated
      conservatively by callers (typically same as ``GENERAL`` with an
      explicit log).
    """

    PERSONAL = "personal"
    COMMAND = "command"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class RoutingStep:
    """One step in a routing plan: activate an adapter and probe keys."""

    adapter_name: str
    keys_to_probe: list[str] = field(default_factory=list)


@dataclass
class RoutingPlan:
    """What to activate and probe for a given query.

    ``intent`` is the single explicit routing axis the chat handler
    dispatches on.  ``steps`` carries the PA probe targets when
    ``intent == PERSONAL``; ``ha_domains`` is observability for the
    HA path.  The legacy ``match_source`` / ``imperative`` fields were
    retired once the if/elif cascade was replaced with intent-keyed
    dispatch — the same information is recoverable from
    ``(bool(steps), bool(ha_domains))`` if any tooling needs it.
    """

    steps: list[RoutingStep] = field(default_factory=list)
    strategy: str = "direct"
    matched_entities: list[str] = field(default_factory=list)
    # HA domains of matched entities/verbs (observability + UI)
    ha_domains: list[str] = field(default_factory=list)
    # Single routing axis the chat handler dispatches on.
    intent: Intent = Intent.UNKNOWN


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

# Terminal punctuation that marks a query as interrogative regardless of
# leading word.  Covers the question-mark glyphs ParaMem-supported scripts
# actually use:
#   ``?``   — ASCII / Latin / most European scripts (U+003F).
#   ``？``  — fullwidth, used in CJK rendering (U+FF1F).
#   ``؟``  — Arabic / Persian / Urdu (U+061F).
# Greek question mark (U+003B, ``;``) is intentionally excluded because it
# is glyphically identical to the ASCII semicolon and would mis-classify
# declarative sentences ending in ``;``.
_INTERROGATIVE_PUNCT = frozenset({"?", "？", "؟"})


def _is_interrogative(text: str, config=None) -> bool:
    """Check if the query is a question (not an imperative command).

    Three-tier detection, in order — each tier produces a definitive
    answer when it can, otherwise falls through to the next:

    1. **Encoder-based classifier** (when ``config`` is a
       :class:`paramem.server.config.SentenceTypeConfig` and the
       encoder + exemplar bank are loaded — the production path).
       Cosine vs multilingual exemplars + margin gate.  Returns the
       encoder's verdict directly when confidence is sufficient.
       Below the margin or on any classifier failure: ``None`` is
       returned by the classifier and we fall through to tier 2.
    2. **Terminal punctuation** — a query ending in any of the
       :data:`_INTERROGATIVE_PUNCT` glyphs is treated as a question
       regardless of leading word.  Language-agnostic and catches
       written queries like ``"Wo wohne ich?"`` / ``"¿Dónde vivo?"``
       / ``"我住在哪里？"``.  Used as the deterministic fallback
       when the encoder isn't available (encoderless boot, tests).
    3. **English first-word lexicon** — for queries that arrive
       without terminal punctuation (some STT engines, conversational
       fragments), match the leading word against
       :data:`_INTERROGATIVE_PREFIXES`.  Possessive contractions
       (``"who's"`` → ``"who"``, ``"what's"`` → ``"what"``) are
       collapsed before lookup.  Only applies when neither tier 1
       nor tier 2 fired.

    The cost asymmetry inside the abstention check (the only consumer)
    favours over-classifying as interrogative: a false positive triggers
    abstention when the user wanted a base-model answer (mildly annoying,
    privacy-safe), a false negative falls through to base-model
    confabulation on personal queries.  Both the encoder fallback path
    and the punctuation tier lean into that bias.

    ``config`` is optional for backward compat with call sites that
    don't have a :class:`ServerConfig` to hand (tests).  ``None``
    skips the encoder tier and uses tiers 2 + 3 only.
    """
    stripped = text.strip()
    if not stripped:
        return False

    # Tier 1: encoder-based classifier (production path).
    if config is not None:
        from paramem.server.sentence_type import (
            SentenceType,
            classify_sentence_type,
        )

        verdict = classify_sentence_type(stripped, config=config)
        if verdict is SentenceType.INTERROGATIVE:
            return True
        if verdict is SentenceType.NON_INTERROGATIVE:
            return False
        # verdict is None — encoder unavailable / margin not met.
        # Fall through to deterministic tiers below.

    # Tier 2: terminal punctuation (language-agnostic).
    if stripped[-1] in _INTERROGATIVE_PUNCT:
        return True
    # Tier 3: English first-word lexicon.
    first_word = stripped.split()[0].lower()
    # Handle possessives: "who's" → "who", "what's" → "what"
    if first_word.endswith("'s"):
        first_word = first_word[:-2]
    return first_word in _INTERROGATIVE_PREFIXES


class QueryRouter:
    """Routes queries to adapters and keys using the knowledge graph.

    Rebuilt after each consolidation.  Indexes are built directly from the
    injected :class:`paramem.training.memory_store.MemoryStore` (canonical
    in both train and simulate modes) and the cumulative knowledge graph.
    Optionally includes an HA entity graph for dual-graph routing.
    """

    def __init__(
        self,
        adapter_dir: Path,
        memory_store,
        graph_path: Path | None = None,
        ha_graph: "HAEntityGraph | None" = None,
        intent_config: "IntentConfig | None" = None,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.graph_path = Path(graph_path) if graph_path else None
        self._ha_graph = ha_graph
        # IntentConfig flows through to classify_intent() so the residual
        # tier (encoder + exemplar bank) gets its margin threshold and
        # fail-closed default.  None means callers receive Intent.UNKNOWN
        # when state signals don't fire — fine for tests that don't care.
        self._intent_config = intent_config
        # The MemoryStore — canonical source of indexed-key entries.
        self._memory_store = memory_store

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
        """Rebuild indexes from the injected :class:`MemoryStore`.

        Walks ``self._memory_store.iter_entries()`` (canonical in both train
        and simulate modes).  Tier ownership comes from the store; no flat
        key→tier reverse lookup needed.

        Also loads from the cumulative knowledge graph when
        :attr:`graph_path` is set.

        Call after consolidation so the indexes reflect the current
        in-memory state.
        """
        self._entity_key_index.clear()
        self._speaker_key_index.clear()
        self._all_entities.clear()

        store = self._memory_store
        if store is not None and len(store) > 0:
            # Walk the store tier-by-tier.  Tier ownership is the store's
            # canonical answer — no flat key→tier reverse lookup needed.
            # ``adapter_id`` is the tier name verbatim, matching what
            # ``model.peft_config`` uses, so probe-time
            # ``switch_adapter(model, step.adapter_name)`` lands on the
            # trained slot.  Do NOT normalize/strip interim stamps:
            # ``episodic_interim_<stamp>`` is the canonical adapter name
            # during an interim window, and the probe-order block in
            # ``route()`` (``*interim_names_sorted, "episodic", ...``)
            # expects unstripped names so the newest interim takes
            # priority over a promoted main "episodic" tier.
            tier_pairs: dict[str, list[dict]] = {}
            for tier_name, key, entry in store.iter_entries():
                tier_pairs.setdefault(tier_name, []).append(
                    {
                        "key": key,
                        "subject": entry.get("subject", ""),
                        "predicate": entry.get("predicate", ""),
                        "object": entry.get("object", ""),
                        "speaker_id": entry.get("speaker_id", ""),
                        "first_seen_cycle": entry.get("first_seen_cycle", 0),
                    }
                )
            for tier_name, pairs in tier_pairs.items():
                self._index_pairs(tier_name, pairs)
        else:
            logger.info("Router reload: memory store empty — indexes will be empty")

        # Also load from graph nodes if available
        if self.graph_path and self.graph_path.exists():
            self._load_graph_entities()

        self._entity_list = sorted(self._all_entities)
        logger.info(
            "Router loaded: %d entities, %d adapters indexed",
            len(self._all_entities),
            len(self._entity_key_index),
        )

    def _index_pairs(self, adapter_name: str, pairs: list[dict]) -> None:
        """Index entity and speaker data from a list of entry dicts.

        Called by :meth:`reload` with pairs projected from
        ``ConsolidationLoop.store`` (the per-tier memory store; canonical
        source for both train and simulate modes).  All sources use the
        same entry shape so the indexing logic is identical:

        * ``subject`` and ``object`` are indexed as entity names (lowercase).
        * ``speaker_id`` is indexed in the per-speaker key set.
        * Predicates of the form ``has_<attr>`` are additionally indexed under
          the de-prefixed attribute name (``"email"``, ``"phone"``, etc.) so
          natural-language attribute queries match the keyed fact directly.

        Args:
            adapter_name: Tier name used as the key in
                ``self._entity_key_index``.
            pairs: List of entry dicts (``key``, ``subject``,
                ``predicate``, ``object``, ``speaker_id``,
                ``first_seen_cycle``).
        """
        index = self._entity_key_index.setdefault(adapter_name, {})
        for kp in pairs:
            key = kp.get("key", "")
            speaker_id = kp.get("speaker_id")
            if speaker_id:
                self._speaker_key_index.setdefault(speaker_id, set()).add(key)
            for field_name in ("subject", "object"):
                entity = kp.get(field_name, "")
                if entity and len(entity) > 1:
                    entity_lower = entity.lower().strip()
                    index.setdefault(entity_lower, set()).add(key)
                    self._all_entities.add(entity_lower)
            # Attribute-predicate indexing: index has_<attr> predicates under
            # the bare attribute name so "what is my email" matches "email"
            # directly and resolves to the has_email key.
            predicate = kp.get("predicate", "")
            if predicate.startswith("has_") and len(predicate) > 4:
                attr_name = predicate[4:]  # strip "has_" prefix
                if attr_name:
                    index.setdefault(attr_name, set()).add(key)
                    self._all_entities.add(attr_name)

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

        # --- Build PA adapter steps (only if PA matched) ---
        steps = []
        if has_pa:
            adapter_keys = self._resolve_keys(pa_entities, allowed_keys)

            # Probe order: procedural → interim newest-first → episodic → semantic → others.
            #
            # 1. procedural first preserves the load-bearing "preferences shape style"
            #    rule (feedback_router_procedural_first.md): behavioral preferences must
            #    surface before any factual context for the PA to feel personalized.
            # 2. interim adapters next because they hold the freshest factual state —
            #    a user correction (move, rename, change of mind) lands in the newest
            #    episodic_interim_<stamp> slot ahead of the next full-cycle merge into
            #    main. Probing interim before main ensures the latest answer wins
            #    instead of returning the stale pre-merge baseline.
            # 3. main episodic: baseline factual snapshot from the last full cycle.
            # 4. semantic: durable but most lossy/abstract — corroboration fallback.
            # 5. others: forward-compat for adapter forms not yet enumerated.
            interim_names_sorted = sorted(
                (n for n in adapter_keys if _interim_sort_key(n) is not None),
                key=lambda n: _interim_sort_key(n) or "",
                reverse=True,
            )

            placed: set[str] = set()
            ordered: list[str] = [
                "procedural",
                *interim_names_sorted,
                "episodic",
                "semantic",
            ]
            # No Top-K slice on the authenticated path.
            #
            # The reachable path here REQUIRES ``allowed_keys`` to be
            # non-empty (line 422: ``if self._entity_key_index and
            # allowed_keys``) — i.e. the speaker is authenticated and
            # owns at least one key.  ``_resolve_keys`` has already
            # intersected with ``allowed_keys`` (line 599), so the per-
            # adapter set is already scoped to the speaker's own facts
            # and entity-narrowed by the query's matched entities.
            #
            # The legacy ``[:MAX_KEYS_PER_QUERY=10]`` slice on
            # ``list(set(...))`` was non-deterministic (Python hash
            # randomization) and caused narrow attribute keys
            # (``has_email`` etc.) to lose a coin-flip on speakers
            # with > 10 entity-touching facts — the failure mode logged
            # in ``project_routing_layer``.  Removing the slice is safe
            # because: (a) the result is already filtered to the
            # speaker's own scope (privacy), (b) ``_resolve_keys`` is
            # entity-narrowed (relevance), (c) when ``inference.preload_cache``
            # is on, downstream cost is O(N) RAM lookups + one
            # ``model.generate`` for reasoning regardless of N.
            # ``sorted(...)`` gives deterministic probe order.
            for adapter_name in ordered:
                if adapter_name in adapter_keys and adapter_name not in placed:
                    steps.append(
                        RoutingStep(
                            adapter_name=adapter_name,
                            keys_to_probe=sorted(adapter_keys[adapter_name]),
                        )
                    )
                    placed.add(adapter_name)

            # Forward-compat: any adapter name we haven't placed (unknown future forms).
            for adapter_name, keys in adapter_keys.items():
                if adapter_name in placed:
                    continue
                steps.append(
                    RoutingStep(
                        adapter_name=adapter_name,
                        keys_to_probe=sorted(keys),
                    )
                )

        strategy = "targeted_probe" if steps else "direct"

        ha_domains = ha_match.domains if ha_match else []

        # --- Intent classification ---
        # State-first dispatch: has_pa → PERSONAL, has_ha → COMMAND.  When
        # neither fires, classify_intent falls through to the encoder
        # residual (cosine vs exemplars + margin gate) or fail-closed.
        # The call is local-cheap when state hits and a single matrix-
        # vector dot product otherwise; failure modes are non-raising.
        from paramem.server.intent import classify_intent  # lazy: breaks router↔intent cycle

        intent = classify_intent(
            text,
            has_graph_match=has_pa,
            has_ha_match=has_ha,
            config=self._intent_config,
        )

        if has_pa or has_ha:
            logger.info(
                "Routed query: intent=%s, pa_entities=%s, ha=%s",
                intent.value,
                pa_entities if has_pa else [],
                ha_domains if has_ha else [],
            )

        return RoutingPlan(
            steps=steps,
            strategy=strategy,
            matched_entities=pa_entities,
            ha_domains=ha_domains,
            intent=intent,
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
