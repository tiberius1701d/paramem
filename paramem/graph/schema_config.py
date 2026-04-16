"""Single source of truth for knowledge-graph taxonomy.

Loads entity types, relation types, and preferred predicates from
configs/schema.yaml. Import-time IO is cached via lru_cache; a YAML
parse error falls back to a hardcoded mirror with a logged error so a
typo cannot brick ``from paramem.graph.schema import Entity`` for the
whole package.

Static type checkers cannot introspect ``Literal[entity_types()]`` —
expected; IDE autocomplete on ``entity.entity_type`` will degrade to
``str``. Acceptable given this codebase is not mypy-strict.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "schema.yaml"

# Hardcoded mirror of configs/schema.yaml.  Updated here whenever the YAML
# gains a new entry; never removed without a schema migration.
_HARDCODED_FALLBACK: dict = {
    "entity_types": {
        "person": {"anchor": "schema:Person"},
        "place": {"anchor": "schema:Place"},
        "organization": {"anchor": "schema:Organization"},
        "event": {"anchor": "schema:Event"},
        "preference": {"anchor": "internal"},
        "concept": {"anchor": "schema:Thing"},
    },
    "fallback_entity_type": "concept",
    "relation_types": ["factual", "temporal", "preference", "social"],
    "fallback_relation_type": "factual",
    "preferred_predicates": [
        {
            "label": "Family/social",
            "items": ["married_to", "parent_of", "child_of", "sibling_of", "has_pet"],
        },
        {
            "label": "Location",
            "items": ["lives_in", "lives_near", "born_in"],
        },
        {
            "label": "Work/education",
            "items": ["works_at", "studies_at", "manages"],
        },
        {
            "label": "Preferences/habits",
            "items": [
                "prefers",
                "likes",
                "dislikes",
                "drinks",
                "eats",
                "listens_to",
                "uses",
                "avoids",
            ],
        },
    ],
    "procedural_entity_types": ["person", "preference"],
    "procedural_predicate_groups": ["Preferences/habits"],
    "anonymizer": {
        "prefixes": [
            {
                "prefix": "Person",
                "entity_type": "person",
                "description": "Person names",
                "primary_for_type": True,
            },
            {
                "prefix": "City",
                "entity_type": "place",
                "description": "City/town names",
                "primary_for_type": True,
            },
            {
                "prefix": "Country",
                "entity_type": "place",
                "description": "Country names",
            },
            {
                "prefix": "Org",
                "entity_type": "organization",
                "description": "Organization names",
                "primary_for_type": True,
            },
            {
                "prefix": "Thing",
                "entity_type": "concept",
                "description": "Other identifying things (apps, products, brands)",
                "primary_for_type": True,
            },
        ]
    },
}


@lru_cache(maxsize=4)
def load_schema_config(path: str | None = None) -> dict:
    """Load and cache the schema YAML.

    Args:
        path: Absolute path to a schema YAML file.  ``None`` resolves to
              ``configs/schema.yaml`` in the project root.

    Returns:
        Parsed YAML as a dict.  Falls back to ``_HARDCODED_FALLBACK`` on
        any IO or parse error, logging a loud error so the failure is not
        silent.
    """
    _REQUIRED_KEYS = frozenset(
        {
            "entity_types",
            "fallback_entity_type",
            "relation_types",
            "fallback_relation_type",
            "preferred_predicates",
            "anonymizer",
        }
    )
    target = Path(path) if path else _DEFAULT_SCHEMA_PATH
    try:
        raw = yaml.safe_load(target.read_text()) or {}
        missing = _REQUIRED_KEYS - raw.keys()
        if missing:
            logger.error(
                "Schema config at %s is missing required keys: %s. "
                "Using hardcoded fallback — taxonomy may be stale.",
                target,
                sorted(missing),
            )
            return _HARDCODED_FALLBACK
        return raw
    except Exception as exc:
        logger.error(
            "Failed to load schema config from %s (%s). "
            "Using hardcoded fallback — taxonomy may be stale.",
            target,
            exc,
        )
        return _HARDCODED_FALLBACK


def reset_cache() -> None:
    """Clear the lru_cache on load_schema_config.

    Required by test fixtures that point the loader at a temporary path.
    """
    load_schema_config.cache_clear()


def entity_types(path: str | None = None) -> tuple[str, ...]:
    """Return all configured entity type names as an ordered tuple.

    Order matches the declaration order in schema.yaml.

    Args:
        path: Optional override path for the schema YAML.
    """
    cfg = load_schema_config(path)
    return tuple(cfg["entity_types"].keys())


def fallback_entity_type(path: str | None = None) -> str:
    """Return the configured fallback entity type name.

    Used when a model-extracted entity type is unrecognised.

    Args:
        path: Optional override path for the schema YAML.
    """
    return load_schema_config(path)["fallback_entity_type"]


def relation_types(path: str | None = None) -> tuple[str, ...]:
    """Return all configured relation type names as an ordered tuple.

    Args:
        path: Optional override path for the schema YAML.
    """
    return tuple(load_schema_config(path)["relation_types"])


def fallback_relation_type(path: str | None = None) -> str:
    """Return the configured fallback relation type name.

    Used when a model-extracted relation type is unrecognised.

    Args:
        path: Optional override path for the schema YAML.
    """
    return load_schema_config(path)["fallback_relation_type"]


def preferred_predicates(path: str | None = None) -> list[dict]:
    """Return the list of preferred-predicate groups.

    Each element is ``{"label": str, "items": list[str]}``.

    Args:
        path: Optional override path for the schema YAML.
    """
    return load_schema_config(path)["preferred_predicates"]


def format_relation_types(path: str | None = None) -> str:
    """Return a comma-separated string of relation type names for prompt injection.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        ``"factual, temporal, preference, social"`` (or whatever the YAML declares).
    """
    return ", ".join(relation_types(path))


def format_entity_types(path: str | None = None, scope: str = "full") -> str:
    """Return a comma-separated string of entity type names for prompt injection.

    Args:
        path: Optional override path for the schema YAML.
        scope: ``"full"`` returns all entity types with fallback annotation;
               ``"procedural"`` returns only the procedural subset.

    Returns:
        ``"person, place, organization, event, preference, concept (fallback — see rule below)"``
        for ``scope="full"``; ``"person, preference"`` for ``scope="procedural"``.
    """
    cfg = load_schema_config(path)
    if scope == "procedural":
        types = list(cfg.get("procedural_entity_types", []))
        return ", ".join(types)
    # full scope
    fb = cfg["fallback_entity_type"]
    types = list(cfg["entity_types"].keys())
    parts = []
    for t in types:
        if t == fb:
            parts.append(f"{t} (fallback — see rule below)")
        else:
            parts.append(t)
    return ", ".join(parts)


def format_predicate_examples(path: str | None = None, scope: str = "full") -> str:
    """Return multi-line predicate example bullets for prompt injection.

    Args:
        path: Optional override path for the schema YAML.
        scope: ``"full"`` returns all groups; ``"procedural"`` filters to
               groups listed in ``procedural_predicate_groups``.

    Returns:
        Newline-joined lines of the form ``"- Family/social: married_to, parent_of, ..."``.
    """
    cfg = load_schema_config(path)
    groups = cfg["preferred_predicates"]
    if scope == "procedural":
        allowed = set(cfg.get("procedural_predicate_groups", []))
        groups = [g for g in groups if g["label"] in allowed]
    lines = [f"- {g['label']}: {', '.join(g['items'])}" for g in groups]
    return "\n".join(lines)


def anonymizer_prefix_to_type(path: str | None = None) -> dict[str, str]:
    """Return ``{prefix_lower: entity_type}`` — reverse map for de-anonymization.

    Keys are lowercased because ``anonymizer_placeholder_pattern`` is
    case-insensitive and the SOTA path uses ``.split("_")[0].lower()``
    before lookup.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Mapping such as ``{"person": "person", "city": "place",
        "country": "place", "org": "organization", "thing": "concept"}``.
    """
    cfg = load_schema_config(path)
    return {
        entry["prefix"].lower(): entry["entity_type"] for entry in cfg["anonymizer"]["prefixes"]
    }


def anonymizer_type_to_prefix(path: str | None = None) -> dict[str, str]:
    """Return ``{entity_type: prefix}`` for entries marked ``primary_for_type=True``.

    Used by the repair path to pick a canonical placeholder when
    re-anonymizing a leaked real name. Only types with a primary prefix
    are eligible; others are treated as out-of-PII-scope.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Mapping such as ``{"person": "Person", "place": "City",
        "organization": "Org", "concept": "Thing"}``.
    """
    cfg = load_schema_config(path)
    return {
        entry["entity_type"]: entry["prefix"]
        for entry in cfg["anonymizer"]["prefixes"]
        if entry.get("primary_for_type", False)
    }


def format_replacement_rules(path: str | None = None) -> str:
    """Return the bulleted Replace: block for the anonymization prompt.

    One line per prefix:
        ``- {description} → {Prefix}_1, {Prefix}_2, ...``

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Newline-joined bullet lines, one per configured prefix.
    """
    cfg = load_schema_config(path)
    lines = []
    for entry in cfg["anonymizer"]["prefixes"]:
        prefix = entry["prefix"]
        desc = entry["description"]
        lines.append(f"- {desc} \u2192 {prefix}_1, {prefix}_2, ...")
    return "\n".join(lines)


def anonymizer_placeholder_pattern(path: str | None = None) -> "re.Pattern[str] | None":
    """Return the compiled placeholder regex, or None when no prefixes are configured.

    Built from the configured prefixes — e.g.
    ``r"^(Person|City|Country|Org|Thing)_\\d+$"``.  Case-insensitive.

    Callers treat ``None`` as "no vocabulary is configured" and branch to the
    appropriate no-vocab behavior — they do not fall through to regex calls.

    When prefixes are present the regex object is returned from Python's
    internal ``re`` compile cache on repeated calls, so there is no
    significant overhead from calling this function multiple times.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        A compiled ``re.Pattern`` matching ``<Prefix>_<digits>``
        (case-insensitive) for all configured prefixes, or ``None`` when
        the prefix list is empty.
    """
    prefixes = [
        p["prefix"] for p in load_schema_config(path).get("anonymizer", {}).get("prefixes", [])
    ]
    if not prefixes:
        return None
    return re.compile(rf"^({'|'.join(prefixes)})_\d+$", re.IGNORECASE)
