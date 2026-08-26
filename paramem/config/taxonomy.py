"""Single source of truth for knowledge-graph taxonomy.

Loads entity types, relation types, and the anonymizer vocabulary from
configs/schema.yaml. Import-time IO is cached via lru_cache. This file is
the ONE declaration of that taxonomy — there is no fallback: an unreadable
file, a YAML parse error, or a missing required key raises ``ValueError``
naming the file and the remediation, at the first reader that touches it.

Static type checkers cannot introspect ``Literal[entity_types()]`` —
expected; IDE autocomplete on ``entity.entity_type`` will degrade to
``str``. Acceptable given this codebase is not mypy-strict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from paramem.utils.identity import canonical
from paramem.utils.paths import find_project_root

_DEFAULT_SCHEMA_PATH = (
    (find_project_root(Path(__file__)) or Path(__file__).resolve().parents[2])
    / "configs"
    / "schema.yaml"
)

_REQUIRED_SCHEMA_KEYS = frozenset(
    {
        "entity_types",
        "fallback_entity_type",
        "relation_types",
        "fallback_relation_type",
        "anonymizer",
    }
)


@lru_cache(maxsize=4)
def load_schema_config(path: str | None = None) -> dict:
    """Load and cache the schema YAML.

    Args:
        path: Absolute path to a schema YAML file.  ``None`` resolves to
              ``configs/schema.yaml`` in the project root.

    Returns:
        Parsed YAML as a dict.  This function is the ONE place every
        reader in this module reaches ``configs/schema.yaml`` through;
        there is no fallback dict.

    Raises:
        ValueError: The file could not be read or parsed, or is missing
            one of the required top-level keys. The message names the
            file path and the remediation.
    """
    target = Path(path) if path else _DEFAULT_SCHEMA_PATH
    try:
        raw = yaml.safe_load(target.read_text()) or {}
    except Exception as exc:
        raise ValueError(
            f"Failed to load schema config from {target} ({exc}). "
            "This file is the one declaration of the knowledge-graph "
            "taxonomy and the anonymizer vocabulary — fix or restore it; "
            "there is no fallback."
        ) from exc
    missing = _REQUIRED_SCHEMA_KEYS - raw.keys()
    if missing:
        raise ValueError(
            f"Schema config at {target} is missing required keys: "
            f"{sorted(missing)}. This file is the one declaration of the "
            "knowledge-graph taxonomy and the anonymizer vocabulary — add "
            "the missing keys; there is no fallback."
        )
    return raw


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


@dataclass(frozen=True)
class ScrubCategory:
    """One local-anonymizer SCAN category: a placeholder prefix, the
    ``sanitization.scrub`` hints it groups, and the span tagger's own
    label vocabulary for it.

    Attributes:
        name: The category's display name — the owning
            ``configs/schema.yaml`` ``anonymizer.prefixes`` row's
            ``prefix``.
        prefix: The placeholder-minting prefix for this category — the
            owning row's ``prefix``.
        hints: This category's configured ``scrub`` hints, in ``scrub``
            order. Never empty — :func:`resolve_scrub_categories` only
            emits a category for a row with at least one configured hint.
        tagger_labels: The owning row's ``tagger_labels`` — the strings
            handed to the span tagger for this category. Never empty —
            :func:`resolve_scrub_categories` refuses a row that is
            activated (at least one hint configured) but declares none.
    """

    name: str
    prefix: str
    hints: tuple[str, ...]
    tagger_labels: tuple[str, ...]


def resolve_scrub_categories(
    scrub: Sequence[str], path: str | None = None
) -> tuple[ScrubCategory, ...]:
    """Group configured ``sanitization.scrub`` hints into SCAN categories.

    A category is a ``configs/schema.yaml`` ``anonymizer.prefixes`` row
    that claims at least one hint present in *scrub* — all of that row's
    claimed hints, in *scrub* order, become one category named after the
    row's prefix, carrying that row's ``tagger_labels`` verbatim. Category
    order is schema row order.

    Six conditions are refused at this door rather than resolved
    arbitrarily or silently under-scrubbing:

    1. A ``scrub_categories`` hint claimed by two distinct prefix rows —
       an ambiguous owner for one hint has no correct answer.
    2. A ``tagger_labels`` label claimed by two distinct prefix rows — a
       tagged span would have two owning categories, and the partition
       the span tagger's caller performs by label would be arbitrary.
    3. A hint present in *scrub* that no row claims — it can carry no
       tagger label, so it would silently scrub nothing.
    4. A row activated by *scrub* (at least one of its ``scrub_categories``
       hints configured) that declares no ``tagger_labels`` — it would
       contribute no label, yield zero tagged spans, and report success.
    5. More than one prefix row setting ``primary_for_type: true`` for the
       same ``entity_type`` — the placeholder-prefix map assumes exactly
       one primary row per entity type.
    6. Two prefix rows whose ``prefix`` values are equal under
       ``casefold()`` — the placeholder tokens each row mints
       (``paramem.cloud.placeholders.mint_placeholder``) would then
       collide under the reply-side rendering equivalence
       (``paramem.cloud.placeholders._rendering_fold``), so the declared
       placeholder vocabulary would not be distinct.

    Args:
        scrub: The configured PII-vocabulary hints
            (``SanitizationConfig.scrub``), in operator order — trusted as
            given, never re-sorted. An empty sequence resolves to an empty
            tuple (the operator opt-out — no category, no scan call).
        path: Optional override path for the schema YAML.

    Returns:
        Categories in schema row order. Never contains a category with
        empty ``hints`` or empty ``tagger_labels``.

    Raises:
        ValueError: On any of the six conditions above.
    """
    cfg = load_schema_config(path)
    prefixes = cfg["anonymizer"]["prefixes"]

    hint_owner: dict[str, str] = {}
    label_owner: dict[str, str] = {}
    primary_owner: dict[str, str] = {}
    prefix_owner: dict[str, int] = {}
    for idx, row in enumerate(prefixes):
        folded_prefix = str(row["prefix"]).casefold()
        owner_idx = prefix_owner.get(folded_prefix)
        if owner_idx is not None and owner_idx != idx:
            raise ValueError(
                f"Prefix {row['prefix']!r} collides under case-folding with "
                f"prefix {prefixes[owner_idx]['prefix']!r} in schema.yaml's "
                "anonymizer.prefixes — the declared placeholder vocabulary "
                "must be distinct under case-folding."
            )
        prefix_owner[folded_prefix] = idx
        for hint in row.get("scrub_categories") or []:
            owner = hint_owner.get(hint)
            if owner is not None and owner != row["prefix"]:
                raise ValueError(
                    f"Scrub hint {hint!r} is claimed by both prefix row "
                    f"{owner!r} and {row['prefix']!r} in schema.yaml's "
                    "anonymizer.prefixes — a hint may be claimed by at most "
                    "one prefix row."
                )
            hint_owner[hint] = row["prefix"]
        for label in row.get("tagger_labels") or []:
            owner = label_owner.get(label)
            if owner is not None and owner != row["prefix"]:
                raise ValueError(
                    f"Tagger label {label!r} is claimed by both prefix row "
                    f"{owner!r} and {row['prefix']!r} in schema.yaml's "
                    "anonymizer.prefixes — a tagger label may be claimed by "
                    "at most one prefix row."
                )
            label_owner[label] = row["prefix"]
        if row.get("primary_for_type", False):
            entity_type = row["entity_type"]
            owner = primary_owner.get(entity_type)
            if owner is not None and owner != row["prefix"]:
                raise ValueError(
                    f"entity_type {entity_type!r} has more than one "
                    f"primary_for_type row in schema.yaml's "
                    f"anonymizer.prefixes: {owner!r} and {row['prefix']!r} — "
                    "only one prefix row per entity_type may set "
                    "primary_for_type: true."
                )
            primary_owner[entity_type] = row["prefix"]

    configured = list(scrub)
    categories: list[ScrubCategory] = []
    covered: set[str] = set()
    for row in prefixes:
        row_claims = set(row.get("scrub_categories") or [])
        if not row_claims:
            continue
        row_hints = tuple(hint for hint in configured if hint in row_claims)
        if not row_hints:
            continue
        tagger_labels = tuple(row.get("tagger_labels") or [])
        if not tagger_labels:
            raise ValueError(
                f"Prefix row {row['prefix']!r} is activated by configured "
                f"scrub hint(s) {row_hints!r} but declares no tagger_labels "
                "in schema.yaml's anonymizer.prefixes — it would contribute "
                "no label to the span tagger and silently scrub nothing."
            )
        categories.append(
            ScrubCategory(
                name=row["prefix"],
                prefix=row["prefix"],
                hints=row_hints,
                tagger_labels=tagger_labels,
            )
        )
        covered.update(row_hints)

    uncovered = [hint for hint in configured if hint not in covered]
    if uncovered:
        raise ValueError(
            f"Configured scrub hint(s) {uncovered!r} are not claimed by any "
            "prefix row in schema.yaml's anonymizer.prefixes — an uncovered "
            "hint can carry no tagger label and would silently scrub "
            "nothing."
        )
    return tuple(categories)


# ---------------------------------------------------------------------------
# prefix <-> entity_type — both directions.
#
# These three functions have no cloud dependency at all — no substitution,
# no resolution map, no LLM round trip — they are purely a projection of
# THIS module's own taxonomy config (``anonymizer.prefixes`` above) onto
# the entity-type vocabulary, so they belong beside the config they read
# rather than beside the placeholder-substitution mechanism. This is also
# what keeps ``paramem/cloud/placeholders.py`` free of any ``paramem.graph``
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

    The minted token shape is BARE (``Person_1``); a braced form (``{Person_1}``)
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
