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
    """One ``configs/schema.yaml`` ``anonymizer.prefixes`` row the
    operator's ``sanitization.scrub`` activates.

    Attributes:
        prefix: The row's placeholder-minting prefix — also the keyword
            the SCAN call must name, folded through
            :func:`~paramem.utils.identity.canonical`, for a value to be
            kept under this row (see
            :func:`~paramem.cloud.anonymize_steps.scan_values`).
    """

    prefix: str


def prefix_descriptions(path: str | None = None) -> tuple[tuple[str, str], ...]:
    """Return every ``anonymizer.prefixes`` row as ``(prefix, description)``
    pairs, in table order — configured or not.

    THE one accessor the SCAN prompt's ``{keywords}`` slot renders from
    (:func:`~paramem.cloud.anonymize_steps.scan_values`) — no second
    reader of the table exists. Every row is included regardless of
    whether the operator's ``sanitization.scrub`` activates it, so the
    model is shown the full keyword vocabulary and never learns which
    kinds are actually scrubbed.

    Args:
        path: Optional override path for the schema YAML.
    """
    cfg = load_schema_config(path)
    return tuple((row["prefix"], row["description"]) for row in cfg["anonymizer"]["prefixes"])


def resolve_scrub_categories(
    scrub: Sequence[str], path: str | None = None
) -> tuple[ScrubCategory, ...]:
    """Resolve configured ``sanitization.scrub`` hints into active rows.

    A row is active when it claims at least one hint present in *scrub* —
    :attr:`ScrubCategory.prefix` names it. Order is schema row order.

    Seven conditions are refused at this door rather than resolved
    arbitrarily or silently under-scrubbing:

    1. A ``scrub_categories`` hint claimed by two distinct prefix rows —
       an ambiguous owner for one hint has no correct answer.
    2. A hint present in *scrub* that no row claims — it can activate
       nothing, so it would silently scrub nothing.
    3. More than one prefix row setting ``primary_for_type: true`` for the
       same ``entity_type`` — the placeholder-prefix map assumes exactly
       one primary row per entity type.
    4. Two prefix rows whose ``prefix`` values collide under
       ``casefold()`` (the mint's rendering equivalence,
       ``paramem.cloud.placeholders._rendering_fold``) — the declared
       placeholder vocabulary would not be distinct. Condition 6's shape
       rule (one word of ASCII letters) makes ``canonical()`` (the keyword
       resolution the SCAN reply is matched against) fold a legal prefix
       identically to ``casefold()`` — no diacritic, blank, or underscore
       for it to collapse — so this one check already covers a
       ``canonical()`` collision between two prefixes too; there is no
       separate arm for it.
    5. A hint whose ``canonical()`` form equals another row's folded
       ``prefix`` — the SCAN reply's keyword-matching step could not tell
       the hint from a real keyword.
    6. A ``prefix`` that is not one word of letters starting with a
       capital — the placeholder shape (``Prefix_N``) and the exact
       folded keyword match both depend on this shape.
    7. An ``entity_type`` not declared under ``entity_types``.

    Args:
        scrub: The configured PII-vocabulary hints
            (``SanitizationConfig.scrub``), in operator order — trusted as
            given, never re-sorted. An empty sequence resolves to an empty
            tuple (the operator opt-out — no category, no scan call).
        path: Optional override path for the schema YAML.

    Returns:
        Active rows in schema row order.

    Raises:
        ValueError: On any of the seven conditions above.
    """
    cfg = load_schema_config(path)
    prefixes = cfg["anonymizer"]["prefixes"]
    declared_entity_types = set(cfg["entity_types"].keys())

    hint_owner: dict[str, str] = {}
    primary_owner: dict[str, str] = {}
    prefix_owner_casefold: dict[str, int] = {}
    prefix_owner_canonical: set[str] = set()
    for idx, row in enumerate(prefixes):
        prefix = str(row["prefix"])
        if not (prefix.isascii() and prefix.isalpha() and prefix[:1].isupper()):
            raise ValueError(
                f"Prefix {prefix!r} in schema.yaml's anonymizer.prefixes is not one "
                "word of letters starting with a capital — the placeholder shape "
                "(Prefix_N) and the SCAN keyword match both depend on this shape."
            )
        entity_type = row["entity_type"]
        if entity_type not in declared_entity_types:
            raise ValueError(
                f"Prefix row {prefix!r} in schema.yaml's anonymizer.prefixes "
                f"declares entity_type {entity_type!r}, which is not one of "
                f"the declared entity_types: {sorted(declared_entity_types)}."
            )
        folded_casefold = prefix.casefold()
        owner_idx = prefix_owner_casefold.get(folded_casefold)
        if owner_idx is not None and owner_idx != idx:
            raise ValueError(
                f"Prefix {prefix!r} collides under case-folding with prefix "
                f"{prefixes[owner_idx]['prefix']!r} in schema.yaml's "
                "anonymizer.prefixes — the declared placeholder vocabulary "
                "must be distinct under case-folding."
            )
        prefix_owner_casefold[folded_casefold] = idx
        # No separate canonical()-collision check here: the shape rule
        # (enforced above) makes canonical() fold a legal prefix
        # identically to casefold(), so a canonical() collision between
        # two prefixes would already have raised as a casefold() collision
        # above (see condition 4's docstring). This set exists only to
        # build all_canon_prefixes for the hint-vs-prefix check below.
        prefix_owner_canonical.add(canonical(prefix))
        for hint in row.get("scrub_categories") or []:
            owner = hint_owner.get(hint)
            if owner is not None and owner != prefix:
                raise ValueError(
                    f"Scrub hint {hint!r} is claimed by both prefix row "
                    f"{owner!r} and {prefix!r} in schema.yaml's "
                    "anonymizer.prefixes — a hint may be claimed by at most "
                    "one prefix row."
                )
            hint_owner[hint] = prefix
        if row.get("primary_for_type", False):
            owner = primary_owner.get(entity_type)
            if owner is not None and owner != prefix:
                raise ValueError(
                    f"entity_type {entity_type!r} has more than one "
                    f"primary_for_type row in schema.yaml's "
                    f"anonymizer.prefixes: {owner!r} and {prefix!r} — "
                    "only one prefix row per entity_type may set "
                    "primary_for_type: true."
                )
            primary_owner[entity_type] = prefix

    # A hint whose canonical() form equals another row's folded prefix
    # would make the SCAN keyword match unable to tell the hint from a
    # real keyword — checked once the full prefix index is built, so a
    # hint equal to its OWN row's prefix (never ambiguous) is not flagged.
    all_canon_prefixes = prefix_owner_canonical
    for row in prefixes:
        prefix = str(row["prefix"])
        for hint in row.get("scrub_categories") or []:
            if canonical(hint) in all_canon_prefixes and canonical(hint) != canonical(prefix):
                raise ValueError(
                    f"Scrub hint {hint!r} on prefix row {prefix!r} in "
                    "schema.yaml's anonymizer.prefixes has the same canonical() "
                    "form as another row's prefix — the SCAN keyword match "
                    "could not tell the hint from a real keyword."
                )

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
        categories.append(ScrubCategory(prefix=row["prefix"]))
        covered.update(row_hints)

    uncovered = [hint for hint in configured if hint not in covered]
    if uncovered:
        raise ValueError(
            f"Configured scrub hint(s) {uncovered!r} are not claimed by any "
            "prefix row in schema.yaml's anonymizer.prefixes — an uncovered "
            "hint would silently scrub nothing."
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

    Closed vocabulary only: :func:`anonymizer_type_to_prefix`
    (schema.yaml's ``primary_for_type`` entries — ``person`` -> ``Person``,
    ``place`` -> ``City``, ``organization`` -> ``Org``, ``concept`` ->
    ``Thing``). THE only place an entity type becomes a placeholder
    prefix; the one production caller passes ``"person"``, which the
    shipped schema always declares a primary row for.

    Raises:
        ValueError: *entity_type* is empty, or no ``primary_for_type`` row
            declares it.
    """
    e = canonical(entity_type)
    closed = anonymizer_type_to_prefix().get(e) if e else None
    if closed is None:
        raise ValueError(
            f"entity_type_to_prefix: no primary_for_type row in schema.yaml's "
            f"anonymizer.prefixes declares entity_type {entity_type!r}."
        )
    return closed


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
