"""Single source of truth for knowledge-graph taxonomy.

Loads entity types, relation types, and the anonymizer vocabulary from
configs/schema.yaml. Import-time IO is cached via lru_cache. This file is
the ONE declaration of that taxonomy — there is no fallback: an unreadable
file, a YAML parse error, or a missing required key raises ``ValueError``
naming the file and the remediation, at the first reader that touches it.

The anonymizer vocabulary is two named lists under ``anonymizer``: ``scrub``
(a row the operator can choose to scrub — the operator's
``sanitization.scrub`` hints activate it, and code replaces its values with
placeholders) and ``allow`` (a row that always reaches the cloud verbatim —
it carries no hints, and code restores every one of its values). Every
reader in this module that walks "the table" walks both lists together,
scrub rows then allow rows.

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
        ValueError: The file could not be read or parsed, is missing one
            of the required top-level keys, or is missing the anonymizer's
            ``scrub`` or ``allow`` list. The message names the file path
            and the remediation.
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
    anonymizer = raw.get("anonymizer") or {}
    missing_lists = {
        name for name in ("scrub", "allow") if not isinstance(anonymizer.get(name), list)
    }
    if missing_lists:
        raise ValueError(
            f"Schema config at {target} is missing anonymizer list(s): "
            f"{sorted(missing_lists)}. The anonymizer vocabulary is two "
            "named lists, `scrub` and `allow` — both must be present (an "
            "empty list is legal); add the missing list(s); there is no "
            "fallback."
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


def _anonymizer_rows(cfg: dict) -> list[tuple[dict, str]]:
    """Return every anonymizer row alongside the name of the list it came from.

    Walks ``anonymizer.scrub`` then ``anonymizer.allow``, each in its own
    declared order — THE one concatenation of the two lists; every reader
    in this module that needs "every row" (scrub or allow) calls this
    instead of re-deriving the order.

    Args:
        cfg: The parsed schema dict, as returned by :func:`load_schema_config`.

    Returns:
        ``(row, list_name)`` pairs, scrub rows first then allow rows,
        where ``list_name`` is ``"scrub"`` or ``"allow"``.
    """
    anonymizer = cfg["anonymizer"]
    return [(row, "scrub") for row in anonymizer["scrub"]] + [
        (row, "allow") for row in anonymizer["allow"]
    ]


def anonymizer_prefix_to_type(path: str | None = None) -> dict[str, str]:
    """Return ``{canonical_prefix: entity_type}`` — reverse map for de-anonymization.

    Spans both anonymizer lists — ``scrub`` and ``allow`` — since a
    placeholder minted for a row in either list must resolve back to its
    entity type. Keys are canonicalized via
    :func:`~paramem.utils.identity.canonical` because the sole lookup site
    (:func:`prefix_to_entity_type`, reached via
    :func:`placeholder_entity_type` for a full placeholder token)
    canonicalizes the placeholder's prefix before querying.  Both sides of
    the map contract use the one identity routine, so a cased or spaced
    YAML prefix can never silently miss.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Mapping such as ``{"person": "person", "city": "place",
        "country": "place", "org": "organization", "profession": "concept"}``.
    """
    cfg = load_schema_config(path)
    return {
        canonical(row["prefix"]): row["entity_type"] for row, _list_name in _anonymizer_rows(cfg)
    }


def anonymizer_type_to_prefix(path: str | None = None) -> dict[str, str]:
    """Return ``{canonical_entity_type: prefix}`` for entries marked ``primary_for_type=True``.

    Used by :func:`entity_type_to_prefix` — the closed-vocabulary lookup
    consulted first when minting a placeholder prefix for an entity type.
    Only types with a primary prefix are eligible; a type with none raises
    (see :func:`entity_type_to_prefix`). ``primary_for_type`` is capped at
    one row per ``entity_type`` across both anonymizer lists — ``scrub``
    and ``allow`` — so a primary declared on either list is found here.

    Keys are canonicalized via :func:`~paramem.utils.identity.canonical`
    to match the canonicalization the lookup site applies to its query.
    Both sides of the map contract use the one identity routine, so a cased
    or spaced YAML ``entity_type`` can never silently miss.

    Args:
        path: Optional override path for the schema YAML.

    Returns:
        Mapping such as ``{"person": "Person", "place": "City",
        "organization": "Org"}``.
    """
    cfg = load_schema_config(path)
    return {
        canonical(row["entity_type"]): row["prefix"]
        for row, _list_name in _anonymizer_rows(cfg)
        if row.get("primary_for_type", False)
    }


@dataclass(frozen=True)
class ScrubCategory:
    """One ``configs/schema.yaml`` ``anonymizer.scrub`` row the operator's
    ``sanitization.scrub`` activates.

    Attributes:
        prefix: The row's placeholder-minting prefix — also the keyword
            the SCAN call must name, folded through
            :func:`~paramem.utils.identity.canonical`, for a value to be
            kept under this row (see
            :func:`~paramem.cloud.anonymize_steps.row_for_keyword`).
    """

    prefix: str


def prefix_descriptions(path: str | None = None) -> tuple[tuple[str, str], ...]:
    """Return every anonymizer row — ``scrub`` rows then ``allow`` rows —
    as ``(prefix, description)`` pairs, each list in its own declared
    order.

    THE one accessor the SCAN prompt's ``{keywords}`` slot renders from
    (:func:`~paramem.cloud.anonymize_steps._render_keywords`) — no second
    reader of the table exists. Every row of both lists is included, so
    the model is shown one combined keyword vocabulary and never learns
    which list a keyword belongs to.

    Args:
        path: Optional override path for the schema YAML.
    """
    cfg = load_schema_config(path)
    return tuple((row["prefix"], row["description"]) for row, _list_name in _anonymizer_rows(cfg))


def resolve_scrub_categories(
    scrub: Sequence[str], path: str | None = None
) -> tuple[ScrubCategory, ...]:
    """Resolve configured ``sanitization.scrub`` hints into active ``scrub``-list rows.

    A row is active when it claims at least one hint present in *scrub* —
    :attr:`ScrubCategory.prefix` names it. Order is ``anonymizer.scrub``
    list order; ``anonymizer.allow`` rows are validated alongside the
    ``scrub`` rows but never activated — an allow row carries no hints, so
    it can never appear in the result.

    The checks below span both lists (a prefix, a ``primary_for_type``
    claim, or a hint can collide across list boundaries); each error names
    ``anonymizer.scrub`` or ``anonymizer.allow``, whichever list the
    offending row is actually in. The following are refused at this door
    rather than resolved arbitrarily or silently under-scrubbing:

    * A row in ``anonymizer.scrub`` with no ``scrub_categories`` — a
      ``scrub`` row with nothing to activate it can never be reached by
      the operator's ``sanitization.scrub`` selection.
    * A row in ``anonymizer.allow`` that carries ``scrub_categories`` —
      an ``allow`` row is defined by reaching the cloud unconditionally;
      hints on it would make it activatable, contradicting the list it is
      declared in.
    * A ``scrub_categories`` hint claimed by two distinct rows — an
      ambiguous owner for one hint has no correct answer.
    * A hint present in *scrub* that no row claims — it can activate
      nothing, so it would silently scrub nothing.
    * More than one row, across both lists, setting ``primary_for_type:
      true`` for the same ``entity_type`` — the placeholder-prefix map
      assumes exactly one primary row per entity type.
    * Two rows, from either list, whose ``prefix`` values collide under
      ``casefold()`` (the mint's rendering equivalence,
      ``paramem.cloud.placeholders._rendering_fold``) — the declared
      placeholder vocabulary would not be distinct. The shape rule below
      (one word of ASCII letters) makes ``canonical()`` (the keyword
      resolution the SCAN reply is matched against) fold a legal prefix
      identically to ``casefold()`` — no diacritic, blank, or underscore
      for it to collapse — so this one check already covers a
      ``canonical()`` collision between two prefixes too; there is no
      separate arm for it.
    * A hint whose ``canonical()`` form equals another row's folded
      ``prefix`` — the SCAN reply's keyword-matching step could not tell
      the hint from a real keyword.
    * A ``prefix`` that is not one word of letters starting with a
      capital — the placeholder shape (``Prefix_N``) and the exact folded
      keyword match both depend on this shape.
    * An ``entity_type`` not declared under ``entity_types``.

    Args:
        scrub: The configured PII-vocabulary hints
            (``SanitizationConfig.scrub``), in operator order — trusted as
            given, never re-sorted. An empty sequence resolves to an empty
            tuple (the operator opt-out — no category, no scan call).
        path: Optional override path for the schema YAML.

    Returns:
        Active rows in ``anonymizer.scrub`` list order.

    Raises:
        ValueError: On any of the conditions above.
    """
    cfg = load_schema_config(path)
    rows = _anonymizer_rows(cfg)
    declared_entity_types = set(cfg["entity_types"].keys())

    hint_owner: dict[str, str] = {}
    primary_owner: dict[str, str] = {}
    prefix_owner_casefold: dict[str, int] = {}
    prefix_owner_canonical: set[str] = set()
    for idx, (row, list_name) in enumerate(rows):
        prefix = str(row["prefix"])
        anon_key = f"anonymizer.{list_name}"
        if not (prefix.isascii() and prefix.isalpha() and prefix[:1].isupper()):
            raise ValueError(
                f"Prefix {prefix!r} in schema.yaml's {anon_key} is not one "
                "word of letters starting with a capital — the placeholder shape "
                "(Prefix_N) and the SCAN keyword match both depend on this shape."
            )
        entity_type = row["entity_type"]
        if entity_type not in declared_entity_types:
            raise ValueError(
                f"Row {prefix!r} in schema.yaml's {anon_key} declares "
                f"entity_type {entity_type!r}, which is not one of the "
                f"declared entity_types: {sorted(declared_entity_types)}."
            )
        row_hints = row.get("scrub_categories") or []
        if list_name == "scrub" and not row_hints:
            raise ValueError(
                f"Row {prefix!r} in schema.yaml's anonymizer.scrub carries no "
                "scrub_categories — every scrub row must serve at least one "
                "sanitization.scrub hint, or it can never be activated."
            )
        if list_name == "allow" and row_hints:
            raise ValueError(
                f"Row {prefix!r} in schema.yaml's anonymizer.allow carries "
                "scrub_categories — an allow row always reaches the cloud "
                "verbatim and takes no hints; move it to anonymizer.scrub "
                "if it should be activatable."
            )
        folded_casefold = prefix.casefold()
        owner_idx = prefix_owner_casefold.get(folded_casefold)
        if owner_idx is not None and owner_idx != idx:
            owner_row, owner_list = rows[owner_idx]
            raise ValueError(
                f"Prefix {prefix!r} in schema.yaml's {anon_key} collides under "
                f"case-folding with prefix {owner_row['prefix']!r} in "
                f"anonymizer.{owner_list} — the declared placeholder vocabulary "
                "must be distinct under case-folding."
            )
        prefix_owner_casefold[folded_casefold] = idx
        # No separate canonical()-collision check here: the shape rule
        # (enforced above) makes canonical() fold a legal prefix
        # identically to casefold(), so a canonical() collision between
        # two prefixes would already have raised as a casefold() collision
        # above. This set exists only to build all_canon_prefixes for the
        # hint-vs-prefix check below.
        prefix_owner_canonical.add(canonical(prefix))
        for hint in row_hints:
            owner = hint_owner.get(hint)
            if owner is not None and owner != prefix:
                raise ValueError(
                    f"Scrub hint {hint!r} is claimed by both row {owner!r} and "
                    f"{prefix!r} in schema.yaml's anonymizer.scrub — a hint may "
                    "be claimed by at most one row."
                )
            hint_owner[hint] = prefix
        if row.get("primary_for_type", False):
            owner = primary_owner.get(entity_type)
            if owner is not None and owner != prefix:
                raise ValueError(
                    f"entity_type {entity_type!r} has more than one "
                    f"primary_for_type row across schema.yaml's anonymizer.scrub "
                    f"and anonymizer.allow: {owner!r} and {prefix!r} — only one "
                    "row per entity_type, across both lists, may set "
                    "primary_for_type: true."
                )
            primary_owner[entity_type] = prefix

    # A hint whose canonical() form equals another row's folded prefix
    # would make the SCAN keyword match unable to tell the hint from a
    # real keyword — checked once the full prefix index is built, so a
    # hint equal to its OWN row's prefix (never ambiguous) is not flagged.
    all_canon_prefixes = prefix_owner_canonical
    for row, list_name in rows:
        prefix = str(row["prefix"])
        for hint in row.get("scrub_categories") or []:
            if canonical(hint) in all_canon_prefixes and canonical(hint) != canonical(prefix):
                raise ValueError(
                    f"Scrub hint {hint!r} on row {prefix!r} in schema.yaml's "
                    f"anonymizer.{list_name} has the same canonical() form as "
                    "another row's prefix — the SCAN keyword match could not "
                    "tell the hint from a real keyword."
                )

    configured = list(scrub)
    categories: list[ScrubCategory] = []
    covered: set[str] = set()
    for row, list_name in rows:
        if list_name != "scrub":
            continue
        row_claims = set(row.get("scrub_categories") or [])
        row_hints = tuple(hint for hint in configured if hint in row_claims)
        if not row_hints:
            continue
        categories.append(ScrubCategory(prefix=row["prefix"]))
        covered.update(row_hints)

    uncovered = [hint for hint in configured if hint not in covered]
    if uncovered:
        raise ValueError(
            f"Configured scrub hint(s) {uncovered!r} are not claimed by any "
            "row in schema.yaml's anonymizer.scrub — an uncovered hint would "
            "silently scrub nothing."
        )
    return tuple(categories)


# ---------------------------------------------------------------------------
# prefix <-> entity_type — both directions.
#
# These three functions have no cloud dependency at all — no substitution,
# no resolution map, no LLM round trip — they are purely a projection of
# THIS module's own taxonomy config (``anonymizer.scrub`` and
# ``anonymizer.allow`` above, together) onto the entity-type vocabulary,
# so they belong beside the config they read
# rather than beside the placeholder-substitution mechanism. This is also
# what keeps ``paramem/cloud/placeholders.py`` free of any ``paramem.graph``
# import: the primitive kit there needs a placeholder SHAPE, never an
# entity-type taxonomy.
# ---------------------------------------------------------------------------


def entity_type_to_prefix(entity_type: str) -> str:
    """Convert an entity-type label to its PascalCase placeholder prefix.

    Closed vocabulary only: :func:`anonymizer_type_to_prefix`
    (schema.yaml's ``primary_for_type`` entries — ``person`` -> ``Person``,
    ``place`` -> ``City``, ``organization`` -> ``Org``). THE only place an
    entity type becomes a placeholder prefix; the one production caller
    passes ``"person"``, which the shipped schema always declares a
    primary row for.

    Raises:
        ValueError: *entity_type* is empty, or no ``primary_for_type`` row
            declares it.
    """
    e = canonical(entity_type)
    closed = anonymizer_type_to_prefix().get(e) if e else None
    if closed is None:
        raise ValueError(
            f"entity_type_to_prefix: no primary_for_type row in schema.yaml's "
            f"anonymizer.scrub or anonymizer.allow declares entity_type {entity_type!r}."
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
