"""Single source of truth for knowledge-graph taxonomy.

Loads entity types and relation types from configs/schema.yaml.
Import-time IO is cached via lru_cache; a YAML
parse error falls back to a hardcoded mirror with a logged error so a
typo cannot brick ``from paramem.graph.schema import Entity`` for the
whole package.

Static type checkers cannot introspect ``Literal[entity_types()]`` —
expected; IDE autocomplete on ``entity.entity_type`` will degrade to
``str``. Acceptable given this codebase is not mypy-strict.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from paramem.utils.identity import canonical
from paramem.utils.paths import find_project_root

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_PATH = (
    (find_project_root(Path(__file__)) or Path(__file__).resolve().parents[2])
    / "configs"
    / "schema.yaml"
)

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
    "relation_types": ["factual", "temporal", "preference", "social", "attribute"],
    "fallback_relation_type": "factual",
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


def anonymizer_prefix_to_type(path: str | None = None) -> dict[str, str]:
    """Return ``{canonical_prefix: entity_type}`` — reverse map for de-anonymization.

    Keys are canonicalized via :func:`~paramem.utils.identity.canonical`
    because the sole lookup site (:func:`prefix_to_entity_type`, reached
    via :func:`placeholder_entity_type` for a full placeholder token)
    canonicalizes the placeholder's prefix before querying.  Both sides of
    the map contract use the one identity routine, so a cased or spaced
    YAML prefix can never silently miss.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Mapping such as ``{"person": "person", "city": "place",
        "country": "place", "org": "organization", "thing": "concept"}``.
    """
    cfg = load_schema_config(path)
    return {
        canonical(entry["prefix"]): entry["entity_type"] for entry in cfg["anonymizer"]["prefixes"]
    }


def anonymizer_type_to_prefix(path: str | None = None) -> dict[str, str]:
    """Return ``{canonical_entity_type: prefix}`` for entries marked ``primary_for_type=True``.

    Used by :func:`entity_type_to_prefix` — the closed-vocabulary lookup
    consulted first when minting a placeholder prefix for an entity type.
    Only types with a primary prefix are eligible; others fall through to
    the open-vocabulary PascalCase path.

    Keys are canonicalized via :func:`~paramem.utils.identity.canonical`
    to match the canonicalization the lookup site applies to its query.
    Both sides of the map contract use the one identity routine, so a cased
    or spaced YAML ``entity_type`` can never silently miss.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Mapping such as ``{"person": "Person", "place": "City",
        "organization": "Org", "concept": "Thing"}``.
    """
    cfg = load_schema_config(path)
    return {
        canonical(entry["entity_type"]): entry["prefix"]
        for entry in cfg["anonymizer"]["prefixes"]
        if entry.get("primary_for_type", False)
    }


# ---------------------------------------------------------------------------
# prefix <-> entity_type — both directions.
#
# These three functions used to live in ``paramem.graph.placeholders``
# alongside the cloud round-trip's placeholder-shape/substitution
# primitives. They moved here (2026-07-21, the ``paramem/cloud/`` package
# carve) because they have no cloud dependency at all — no substitution, no
# resolution map, no LLM round trip — they are purely a projection of THIS
# module's own taxonomy config (``anonymizer.prefixes`` above) onto the
# entity-type vocabulary, so they belong beside the config they read rather
# than beside the placeholder-substitution mechanism. This is also what
# keeps ``paramem/cloud/placeholders.py`` free of any ``paramem.graph``
# import: the primitive kit there needs a placeholder SHAPE, never an
# entity-type taxonomy.
# ---------------------------------------------------------------------------


def entity_type_to_prefix(entity_type: str) -> str:
    """Convert an entity-type label to its PascalCase placeholder prefix.

    Closed vocabulary first: :func:`anonymizer_type_to_prefix`
    (schema.yaml's ``primary_for_type`` entries — ``person`` -> ``Person``,
    ``place`` -> ``City``, ``organization`` -> ``Org``, ``concept`` ->
    ``Thing``). Any other label is PascalCase-joined on whitespace /
    hyphen / underscore — ``"work_of_art"`` -> ``"WorkOfArt"``,
    ``"language"`` -> ``"Language"``. Empty / whitespace-only input, or a
    type that PascalCase-joins to nothing, falls back to ``"Entity"``.

    THE only place an entity type becomes a placeholder prefix.
    """
    if not entity_type:
        return "Entity"
    e = canonical(entity_type)
    if not e:
        return "Entity"
    closed = anonymizer_type_to_prefix().get(e)
    if closed is not None:
        return closed
    # ``canonical`` has already folded ``_`` and whitespace runs to single
    # spaces, so only ``-`` (which it preserves verbatim) still separates
    # words here.  Splitting on ``[\s_\-]+`` again would re-apply a fold that
    # has already run.
    pascal = "".join(p.capitalize() for p in e.replace("-", " ").split())
    return pascal or "Entity"


def prefix_to_entity_type(prefix: str) -> str:
    """Convert a placeholder prefix back to an entity-type label.

    Closed vocabulary first: :func:`anonymizer_prefix_to_type` (``city`` ->
    ``place``, ``org`` -> ``organization``, ...). Open vocabulary: any
    other prefix names its own type — the cloud round trip's
    brace-binding protocol mints a prefix that IS the type name for a
    novel entity (``Project_1``, ``Paper_1``, ``Language_1``), so the
    derived type passes through rather than being treated as unrecognised.
    Falls back to ``"concept"`` only when ``prefix`` itself is empty.

    THE only place a placeholder prefix resolves to an entity type.
    """
    p = canonical(prefix or "")
    return anonymizer_prefix_to_type().get(p) or p or "concept"


def placeholder_entity_type(token: str) -> str:
    """Convert a placeholder TOKEN (e.g. ``"Person_1"``) to its entity-type
    label — brace-tolerant.

    THE single site that derives an entity type from a placeholder TOKEN
    (as opposed to an already-isolated prefix string, which
    :func:`prefix_to_entity_type` handles). Strips a surrounding ``{...}``
    shape before splitting the prefix off the token, then delegates to
    :func:`prefix_to_entity_type`.

    Token shape today is BARE (``Person_1``); a braced form (``{Person_1}``)
    exists only for the in-text detection net
    (:data:`~paramem.cloud.placeholders.PLACEHOLDER_TOKEN_RE`) and the
    cloud round trip's own brace-binding mint protocol. Bypassing the
    brace strip here would silently mistype a braced token at a future
    format flip: ``"{Person_1}".split("_")[0]`` is ``"{Person"``, which is
    not in the closed vocabulary and passes through open-vocabulary as its
    own (wrong) type ``"{person"`` — corrupting every live consumer of the
    derived type: Cloud-minted-entity type inference in
    :func:`~paramem.graph.relation_build.apply_rebuild` and entity-surface
    correction in :mod:`paramem.graph.entity_correction`. Stripping braces
    first means this function survives that flip unchanged.
    """
    t = (token or "").strip()
    if len(t) >= 2 and t[0] == "{" and t[-1] == "}":
        t = t[1:-1]
    prefix = t.split("_", 1)[0] if "_" in t else t
    return prefix_to_entity_type(prefix)
