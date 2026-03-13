"""Entity profile builder for entity-keyed natural language memory.

Converts knowledge graph relations into entity-keyed QA pairs with
temporal anchoring. Profiles are expressed as natural language — the
one format proven to work at 81% recall. The graph is a transient
intermediate for structural operations; the adapter only ever sees
QA pairs.
"""

import logging

import networkx as nx

from paramem.graph.qa_generator import _QA_TEMPLATES

logger = logging.getLogger(__name__)

# Sentence templates for converting triples to natural language.
# Keyed by normalized predicate. Falls back to generic.
_SENTENCE_TEMPLATES = {
    "lives_in": "{subject} lives in {object}.",
    "works_at": "{subject} works at {object}.",
    "works_as": "{subject} works as {object}.",
    "has_pet": "{subject} has a pet: {object}.",
    "prefers": "{subject} prefers {object}.",
    "studies_at": "{subject} studied at {object}.",
    "speaks": "{subject} speaks {object}.",
    "knows": "{subject} knows {object}.",
    "has_hobby": "{subject} enjoys {object}.",
    "manages": "{subject} manages {object}.",
    "has_age": "{subject} is {object}.",
    "born_on": "{subject}'s birthday is {object}.",
    "uses": "{subject} uses {object}.",
    "likes": "{subject} likes {object}.",
    "favorite": "{subject}'s favorite is {object}.",
    "visited": "{subject} visited {object}.",
    "read": "{subject} read {object}.",
    "fixed": "{subject} fixed {object}.",
    "attended": "{subject} attended {object}.",
    "bought": "{subject} bought {object}.",
    "watched": "{subject} watched {object}.",
    "cooked": "{subject} cooked {object}.",
    "debugged": "{subject} debugged {object}.",
    "presented": "{subject} presented {object}.",
    "started": "{subject} started {object}.",
    "collaborates_with": "{subject} collaborates with {object}.",
}

# Question variants for entity profiles
_PROFILE_QUESTIONS = [
    "What do you know about {entity}?",
    "Tell me about {entity}.",
]


def _normalize_predicate(predicate: str) -> str:
    """Normalize a predicate for template lookup."""
    return predicate.strip().lower().replace(" ", "_").replace("-", "_")


def _relation_to_sentence(
    relation: dict,
    timestamp: str | None = None,
) -> str:
    """Convert a single relation to a natural language sentence.

    Args:
        relation: Dict with subject, predicate, object keys.
        timestamp: Optional temporal anchor (e.g. "On March 5, 2026").
    """
    subject = relation["subject"]
    predicate = relation["predicate"]
    obj = relation["object"]

    norm_pred = _normalize_predicate(predicate)
    template = _SENTENCE_TEMPLATES.get(norm_pred)

    if template:
        sentence = template.format(subject=subject, object=obj)
    else:
        readable = predicate.replace("_", " ")
        sentence = f"{subject} {readable} {obj}."

    if timestamp:
        sentence = f"{timestamp}, {sentence[0].lower()}{sentence[1:]}"

    return sentence


def _get_temporal_prefix(edge_data: dict) -> str | None:
    """Build a temporal prefix from graph edge metadata.

    Uses last_seen session ID to anchor facts temporally.
    Returns None if no useful temporal data exists.
    """
    sessions = edge_data.get("sessions", [])
    recurrence = edge_data.get("recurrence_count", 1)

    if not sessions:
        return None

    last_session = sessions[-1]

    if recurrence > 1:
        return f"As of {last_session}"
    return f"In {last_session}"


def _get_entity_relations(
    graph: nx.MultiDiGraph,
    entity_name: str,
) -> list[dict]:
    """Get all relations where entity is the subject.

    Returns list of dicts with subject, predicate, object, and edge metadata.
    """
    relations = []
    if entity_name not in graph:
        return relations

    for _, target, data in graph.edges(entity_name, data=True):
        relations.append(
            {
                "subject": entity_name,
                "predicate": data.get("predicate", "related_to"),
                "object": target,
                "sessions": data.get("sessions", []),
                "recurrence_count": data.get("recurrence_count", 1),
                "first_seen": data.get("first_seen", ""),
                "last_seen": data.get("last_seen", ""),
            }
        )

    return relations


def build_entity_profile(
    entity_name: str,
    graph: nx.MultiDiGraph,
    include_timestamps: bool = True,
    max_relations: int = 15,
) -> str:
    """Build a natural language profile for an entity from the graph.

    Groups all relations where the entity is the subject and converts
    them to natural language sentences with optional temporal anchoring.

    Args:
        entity_name: The entity to profile.
        graph: The cumulative knowledge graph.
        include_timestamps: Whether to include temporal anchoring.
        max_relations: Maximum relations to include (cap for token budget).

    Returns:
        Natural language profile text, or empty string if no relations.
    """
    relations = _get_entity_relations(graph, entity_name)
    if not relations:
        return ""

    # Prioritize by recurrence (most reinforced facts first)
    relations.sort(key=lambda r: r.get("recurrence_count", 1), reverse=True)
    if len(relations) > max_relations:
        relations = relations[:max_relations]

    sentences = []
    for rel in relations:
        timestamp = None
        if include_timestamps:
            timestamp = _get_temporal_prefix(rel)
        sentence = _relation_to_sentence(rel, timestamp=timestamp)
        sentences.append(sentence)

    return " ".join(sentences)


def profile_to_qa_pairs(
    entity_name: str,
    profile_text: str,
    relations: list[dict],
    num_variants: int = 1,
    max_profile_relations: int = 5,
    graph: nx.MultiDiGraph | None = None,
) -> list[dict]:
    """Generate QA pairs from an entity profile.

    Produces:
    1. Profile QA: "What do you know about {entity}?" -> short profile (top N facts)
    2. Fact-specific QA: "Where does {entity} work?" -> single fact (all relations)

    The broad profile answer is capped at max_profile_relations to prevent
    long answers from dominating the gradient signal over concise fact QA.

    Args:
        entity_name: The entity name.
        profile_text: The NL profile from build_entity_profile().
        relations: The entity's relations (for fact-specific QA).
        num_variants: Number of profile question variants to generate.
        max_profile_relations: Max relations in the broad profile answer.
        graph: Optional graph for building capped profile. If None, uses profile_text as-is.

    Returns:
        List of QA dicts with 'question' and 'answer' keys.
    """
    if not profile_text:
        return []

    qa_pairs = []

    # Build a shorter profile for the broad answer (cap to top N by recurrence)
    if graph is not None:
        short_profile = build_entity_profile(
            entity_name, graph, max_relations=max_profile_relations
        )
    else:
        short_profile = profile_text

    # Profile-level QA with variants
    for i, q_template in enumerate(_PROFILE_QUESTIONS):
        if i >= num_variants:
            break
        qa_pairs.append(
            {
                "question": q_template.format(entity=entity_name),
                "answer": short_profile,
            }
        )

    # Fact-specific QA from templates
    for rel in relations:
        norm_pred = _normalize_predicate(rel["predicate"])
        templates = _QA_TEMPLATES.get(norm_pred)
        if templates:
            for q_template, a_template in templates:
                qa_pairs.append(
                    {
                        "question": q_template.format(subject=rel["subject"], object=rel["object"]),
                        "answer": a_template.format(subject=rel["subject"], object=rel["object"]),
                    }
                )

    return qa_pairs


def build_all_entity_qa(
    graph: nx.MultiDiGraph,
    entity_names: list[str],
    include_timestamps: bool = True,
    max_relations_per_entity: int = 15,
    num_variants: int = 1,
    max_profile_relations: int = 5,
) -> list[dict]:
    """Build QA pairs for all entities in the registry.

    This is the main entry point for the consolidation loop.

    Args:
        graph: The cumulative knowledge graph.
        entity_names: List of entity names to profile.
        include_timestamps: Whether to include temporal anchoring.
        max_relations_per_entity: Cap relations per entity.
        num_variants: Profile question variants per entity.
        max_profile_relations: Max relations in broad profile answer.

    Returns:
        Flat list of QA dicts ready for training.
    """
    all_qa = []
    for entity_name in entity_names:
        relations = _get_entity_relations(graph, entity_name)
        if not relations:
            continue

        profile = build_entity_profile(
            entity_name,
            graph,
            include_timestamps=include_timestamps,
            max_relations=max_relations_per_entity,
        )
        if not profile:
            continue

        qa_pairs = profile_to_qa_pairs(
            entity_name,
            profile,
            relations,
            num_variants=num_variants,
            max_profile_relations=max_profile_relations,
            graph=graph,
        )
        all_qa.extend(qa_pairs)

    logger.info(
        "Built %d QA pairs for %d entities",
        len(all_qa),
        len(entity_names),
    )
    return all_qa
