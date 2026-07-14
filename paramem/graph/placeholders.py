"""The anonymize <-> deanonymize placeholder contract, whole, in one module.

Every extraction path that touches PII goes through this module's
primitives: mint a placeholder token, build the real-name <-> placeholder
table, detect declared/shaped tokens in text, resolve core + SOTA-sourced
tables into one substitution map, substitute, and de-anonymize. Nothing
here has an opinion about WHERE the table comes from (local anonymizer,
SOTA enrichment, cloud-egress helper) or WHAT happens to the result
(SOTA prompt construction, plausibility filtering, Relation
construction) — those are extraction-pipeline concerns that live in
:mod:`paramem.graph.extractor`.

Token shape today is BARE (``Person_1``); a braced form (``{Person_1}``)
exists only for the in-text detection net and for SOTA's own
brace-binding mint protocol. Nothing here hardcodes the bare shape as
load-bearing — a future format flip only touches :func:`mint_placeholder`
and :data:`PLACEHOLDER_SHAPE_RE`.

Load-bearing invariants preserved from the pre-refactor implementation:

* ``observed is None`` in :func:`_resolution_map` means CORE UNSCOPED —
  never ``set()`` (an empty set would scope CORE to nothing and drop
  every fact on the paths that pass ``None``).
* CORE-LAST precedence: :func:`_resolution_map` merges ``sota_bindings``
  first, then ``.update()``s the CORE ``reverse`` map on top — CORE
  always wins a key collision.
* :func:`_apply_bindings` is the single deanon exit gate: predicate
  invariant (drop, never repair) BEFORE substitution, then substitute
  subject/object, then a residual sweep over every ``_FACT_FIELDS``
  field. The two drop categories are returned already partitioned.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from paramem.graph.name_match import is_speaker_id
from paramem.graph.schema import Entity, SessionGraph
from paramem.graph.schema_config import anonymizer_prefix_to_type, anonymizer_type_to_prefix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shape — the ONE placeholder-shape source, serving both the anchored
# full-string validator and the unanchored in-text detector.
# ---------------------------------------------------------------------------

# PascalCase prefix + "_" + positive integer. The prefix vocabulary is
# open: "Person" / "City" / "Org" / "Thing" are common (configured in
# configs/schema.yaml) but any PascalCase noun is well-formed — a model
# is free to mint "University_1" / "Project_1" / "Language_1" for a
# type none of the common prefixes fit.
_BARE_PLACEHOLDER_SHAPE = r"[A-Z][A-Za-z]*_\d+"

# Anchored full-string validator — used by table normalize/validate
# (:func:`_normalize_anonymization_mapping`, :func:`_mapping_is_canonical`)
# to classify whether a table ENTRY (not embedded in surrounding text) is
# placeholder-shaped. Collapses the previous three-way split between
# schema_config's `_UNIVERSAL_PLACEHOLDER_RE` / `anonymizer_placeholder_pattern()`,
# extractor's `_PLACEHOLDER_TOKEN_RE`, and a test-local `mint_re`.
PLACEHOLDER_SHAPE_RE = re.compile(rf"^{_BARE_PLACEHOLDER_SHAPE}$")

# Unanchored in-text detector — matches EITHER a braced mint
# (``{Event_1}``, prefix laxly ``\w+`` since SOTA's own mint protocol is
# the only producer of the braced form and doesn't always comply with
# PascalCase) OR a bare token (``Person_2``, strict PascalCase,
# word-boundary anchored). Used to scan facts/transcript text for
# placeholder tokens: :func:`_check_mapping_totality`'s orphan scan, the
# residual sweep in :func:`_apply_bindings`, and the ``observed``
# legality-domain scan in :func:`~paramem.graph.extractor._sota_pipeline`.
PLACEHOLDER_TOKEN_RE = re.compile(rf"\{{(\w+_\d+)\}}|\b({_BARE_PLACEHOLDER_SHAPE})\b")


# ---------------------------------------------------------------------------
# Mint — the only place a placeholder token string is built.
# ---------------------------------------------------------------------------


def mint_placeholder(existing_values: Iterable[object], prefix: str) -> str:
    """Mint a fresh, non-colliding placeholder token for ``prefix``.

    Scans ``existing_values`` (typically a mapping's ``.values()``) for
    existing ``Prefix_N`` tokens sharing ``prefix`` and returns
    ``f"{prefix}_{max(N) + 1}"``. Scanning (rather than a local counter
    that starts at 1 and is blind to what's already in the table) is
    what makes this safe against a prefix already used by an
    LLM-emitted hint — PROVIDED the caller includes that hint's values
    in ``existing_values`` at mint time. A caller that scans only its
    own table and merges an LLM hint in afterwards is not protected by
    this function; every call site in this module passes the union of
    both sources.

    THE only place a placeholder token string is built. Collapses three
    previous implementations: the anonymizer's per-prefix counter
    closure inside ``_build_anonymization_mapping`` (counted from 1,
    blind to the table), that same function's already-scanning
    speaker-name-fallback mint, and the leak-repair mint inside
    ``_repair_anonymization_leaks`` (its own scanning closure,
    duplicating this one).
    """
    max_n = 0
    for v in existing_values:
        if not isinstance(v, str):
            continue
        if v.startswith(f"{prefix}_"):
            tail = v.split("_")[-1]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
    return f"{prefix}_{max_n + 1}"


def braced(token: str) -> str:
    """Wrap a bare placeholder token in braces: ``"Person_1"`` -> ``"{Person_1}"``.

    THE only place a placeholder token is wrapped in braces. Used both
    to build the braced-literal substitution key in :func:`_apply_bindings`
    and to construct SOTA's braced-mint transcript notation in
    :func:`~paramem.graph.extractor._reconstruct_updated_transcript`.
    """
    return f"{{{token}}}"


# ---------------------------------------------------------------------------
# Substitution — word-boundary text substitution over a {key: value} table.
# ---------------------------------------------------------------------------


def _is_word_char(c: str) -> bool:
    """Match Python regex ``\\w`` semantics: alphanumeric (Unicode-aware) or underscore."""
    return c.isalnum() or c == "_"


def _substitute_whole_words(
    text: str,
    mapping: dict[str, str],
) -> str:
    """Replace whole-word occurrences of mapping keys with their values.

    Mirrors the previous ``for k, v in mapping: text = re.sub(rf"\\b{re.escape(k)}\\b", v, text)``
    pattern in a single token-walk pass.  Boundaries follow the ``\\b``
    semantics: word-character/non-word-character transitions, where word-
    character means ``c.isalnum() or c == "_"`` (Unicode-aware).

    Longest keys are tried first at each position so multi-word keys
    preempt single-word prefixes (``"Person_2"`` before ``"Person"``).
    Empty / non-string keys are skipped defensively — local extractors
    occasionally emit ``null`` mapping entries.  Matching is
    case-sensitive — every call site's mapping keys are exact-case
    entity names or placeholder tokens.

    Matching is EXACT — never case-, separator-, or diacritic-folded. Keys
    here are literal surfaces: real names in the ANONYMIZE direction and
    machine-minted placeholder tokens in the DEANONYMIZE direction. Folding
    would let a mapped person name silently consume its lowercase
    common-noun homograph (a person named "Bill" matching the electricity
    "bill"), and would let literal placeholder text resolve against a real
    name, defeating the fail-closed residual-token drop (b14a880) before it
    ever sees the token. Identity reconciliation — matching a mapping key
    to the fold graph's own canonical node-key text — is a separate step
    performed by the one caller that needs it, before this function ever
    sees the mapping — see :meth:`~paramem.training.consolidation.
    ConsolidationLoop._run_graph_enrichment`.
    """
    if not mapping or not text:
        return text
    normalized = {k: v for k, v in mapping.items() if isinstance(k, str) and k}
    if not normalized:
        return text
    keys_sorted = sorted(normalized, key=len, reverse=True)

    parts: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        if not _is_word_char(text[pos]):
            parts.append(text[pos])
            pos += 1
            continue
        matched = False
        for key in keys_sorted:
            klen = len(key)
            end = pos + klen
            if end > n:
                continue
            if end < n and _is_word_char(text[end]):
                continue
            slice_ = text[pos:end]
            if slice_ != key:
                continue
            replacement = normalized[key]
            if not isinstance(replacement, str):
                continue
            parts.append(replacement)
            pos = end
            matched = True
            break
        if not matched:
            j = pos + 1
            while j < n and _is_word_char(text[j]):
                j += 1
            parts.append(text[pos:j])
            pos = j
    return "".join(parts)


# ---------------------------------------------------------------------------
# prefix <-> entity_type — both directions.
# ---------------------------------------------------------------------------


def entity_type_to_prefix(entity_type: str) -> str:
    """Convert an entity-type label to its PascalCase placeholder prefix.

    Closed vocabulary first: :func:`~paramem.graph.schema_config.
    anonymizer_type_to_prefix` (schema.yaml's ``primary_for_type``
    entries — ``person`` -> ``Person``, ``place`` -> ``City``,
    ``organization`` -> ``Org``, ``concept`` -> ``Thing``). Any other
    label is PascalCase-joined on whitespace / hyphen / underscore —
    ``"work_of_art"`` -> ``"WorkOfArt"``, ``"language"`` -> ``"Language"``.
    Empty / whitespace-only input, or a type that PascalCase-joins to
    nothing, falls back to ``"Entity"``.

    THE only place an entity type becomes a placeholder prefix.
    Collapses two implementations that disagreed on the open-vocabulary
    fallback: the anonymizer table builder used ``str.capitalize()``
    (``"work_of_art"`` -> ``"Work_of_art"``); the leak-repair path used
    PascalCase-joining via a hardcoded ``_TYPE_PREFIX_OVERRIDES`` dict
    that duplicated the same four schema-config entries. PascalCase
    wins — no compat flag.
    """
    if not entity_type:
        return "Entity"
    e = entity_type.strip().lower()
    if not e:
        return "Entity"
    closed = anonymizer_type_to_prefix().get(e)
    if closed is not None:
        return closed
    parts = re.split(r"[\s_\-]+", e)
    pascal = "".join(p.capitalize() for p in parts if p)
    return pascal or "Entity"


def prefix_to_entity_type(prefix: str) -> str:
    """Convert a placeholder prefix back to an entity-type label.

    Closed vocabulary first: :func:`~paramem.graph.schema_config.
    anonymizer_prefix_to_type` (``city`` -> ``place``, ``org`` ->
    ``organization``, ...). Open vocabulary: any other prefix names its
    own type — SOTA's brace-binding protocol mints a prefix that IS the
    type name for a novel entity (``Project_1``, ``Paper_1``,
    ``Language_1``), so the derived type passes through rather than
    being treated as unrecognised. Falls back to ``"concept"`` only when
    ``prefix`` itself is empty.

    THE only place a placeholder prefix resolves to an entity type.
    Collapses two implementations that disagreed on policy: the
    entity-rebuild loop in :func:`~paramem.graph.extractor._sota_pipeline`
    already derived the open-vocabulary type from the prefix;
    :func:`~paramem.graph.entity_correction.correct_entity_surfaces`
    instead skipped the placeholder entirely on an unrecognised prefix
    (closed vocabulary). The open policy wins, matching the
    entity-rebuild loop's existing behaviour.
    """
    p = (prefix or "").strip().lower()
    return anonymizer_prefix_to_type().get(p) or p or "concept"


def placeholder_entity_type(token: str) -> str:
    """Convert a placeholder TOKEN (e.g. ``"Person_1"``) to its entity-type
    label — brace-tolerant.

    THE single site that derives an entity type from a placeholder TOKEN
    (as opposed to an already-isolated prefix string, which
    :func:`prefix_to_entity_type` handles). Strips a surrounding
    ``{...}`` shape before splitting the prefix off the token, then
    delegates to :func:`prefix_to_entity_type`.

    Collapses three previously-duplicated inline
    ``prefix_to_entity_type(placeholder.split("_")[0])`` derivations
    (``paramem/training/consolidation.py``, ``paramem/graph/extractor.py``,
    ``paramem/graph/entity_correction.py``). Token shape today is BARE
    (``Person_1``); a braced form (``{Person_1}``) exists only for the
    in-text detection net and SOTA's own brace-binding mint protocol (see
    the module docstring). Bypassing the brace strip here would silently
    mistype a braced token at a future format flip: ``"{Person_1}".split
    ("_")[0]`` is ``"{Person"``, which is not in the closed vocabulary and
    passes through open-vocabulary as its own (wrong) type ``"{person"`` —
    never matching any ``pii_scope`` membership check and silently
    unmasking the entity. Stripping braces first means this function
    survives that flip unchanged.
    """
    t = (token or "").strip()
    if len(t) >= 2 and t[0] == "{" and t[-1] == "}":
        t = t[1:-1]
    prefix = t.split("_", 1)[0] if "_" in t else t
    return prefix_to_entity_type(prefix)


# ---------------------------------------------------------------------------
# Table normalize / validate — ONE normalizer, ONE validator, shared by the
# CORE anonymizer table and the SOTA `bindings` table.
# ---------------------------------------------------------------------------


def _normalize_anonymization_mapping(
    mapping: dict, *, placeholder_side: str = "value"
) -> tuple[dict, dict]:
    """Normalize a table to canonical direction — placeholder shape on
    ``placeholder_side``.

    ``placeholder_side="value"`` (default) — the CORE anonymizer table,
    canonical direction ``{real_name: placeholder}``.
    ``placeholder_side="key"`` — the SOTA ``bindings`` table, canonical
    direction ``{placeholder: real_text}`` — the OPPOSITE direction,
    since a binding maps a placeholder SOTA minted to the real span it
    stands for.

    Per-entry classification: whichever side of the pair matches
    :data:`PLACEHOLDER_SHAPE_RE` becomes ``placeholder_side`` in the
    output. When BOTH sides match (a real-world name that happens to be
    PascalCase_N-shaped, e.g. ``GPT_4``, ``COVID_19``), the caller's
    declared ``placeholder_side`` breaks the tie — the entry is kept
    as-is, not dropped. Only NEITHER side matching is genuinely
    ambiguous and dropped (logging).

    Returns ``(canonical_mapping, stats)`` where ``stats`` has
    ``{inverted, dropped}`` counts — surfaces the mapping-quality signal
    to callers so they can persist it in diagnostics (ambiguous-drop can
    otherwise silently void real entities or SOTA-minted entries).

    THE only normalizer for either table — previously the CORE table had
    a normalizer and the SOTA ``bindings`` table had none at all (an
    inverted binding like ``{"Acme": "Org_9"}`` passed straight through
    into the substitution map).
    """
    if not mapping:
        return mapping, {"inverted": 0, "dropped": 0}
    out: dict = {}
    inverted = 0
    dropped = 0
    for k, v in mapping.items():
        k_match = bool(PLACEHOLDER_SHAPE_RE.match(str(k)))
        v_match = bool(PLACEHOLDER_SHAPE_RE.match(str(v)))
        if placeholder_side == "key":
            wants_invert = v_match and not k_match
            # Both sides matching the shape (e.g. a binding onto a
            # real-world name that also happens to be PascalCase_N-shaped,
            # like "GPT_4") is a genuine tie, not an unresolvable
            # ambiguity: the caller already told us which side is
            # DECLARED as the placeholder via `placeholder_side`, so trust
            # it rather than dropping the entry.
            keep_as_is = k_match
        else:
            wants_invert = k_match and not v_match
            keep_as_is = v_match
        if wants_invert:
            out[v] = k
            inverted += 1
        elif keep_as_is:
            out[k] = v
        else:
            # Neither side matches the placeholder shape — we cannot tell
            # which side is the placeholder. Dropping is safer than
            # keeping: retaining would corrupt the resolution map with a
            # real-to-real entry.
            dropped += 1
    if inverted:
        logger.info(
            "Anonymization table: inverted %d/%d pair(s) to canonical "
            "direction (placeholder_side=%r)",
            inverted,
            len(mapping),
            placeholder_side,
        )
    if dropped:
        logger.warning(
            "Anonymization table: dropped %d/%d ambiguous pair(s) (both or "
            "neither side matches the placeholder shape); affected entries "
            "will not resolve.",
            dropped,
            len(mapping),
        )
    return out, {"inverted": inverted, "dropped": dropped}


def _mapping_is_canonical(mapping: dict, *, placeholder_side: str = "value") -> bool:
    """Validate the structural contract: shape, direction.

    ``placeholder_side`` selects which side of the table must carry the
    placeholder shape — ``"value"`` (default) for the CORE table
    (``{real_name: placeholder}``), ``"key"`` for SOTA ``bindings``
    (``{placeholder: real_text}``).

    Returns ``True`` iff:

    * **Shape** — every entry on ``placeholder_side`` matches
      :data:`PLACEHOLDER_SHAPE_RE`. The prefix vocabulary is open:
      ``Person`` / ``City`` / ``Org`` / ``Thing`` are common but not
      exhaustive.
    * **Direction** — no entry on the OTHER side matches the placeholder
      shape (would be self-mapped).

    **Uniqueness is deliberately NOT a property of this table — do not
    re-add it.** The CORE forward map is many-to-one BY CONSTRUCTION:
    :func:`_build_anonymization_mapping` folds an entity's PII attribute
    values onto that entity's own placeholder (``placeholders.py:877``),
    so two entries legitimately collapsing onto the same
    ``placeholder_side`` value is the normal, intended shape of the
    table — not a defect to detect here. A former uniqueness clause
    (``len(set(placeholder_entries)) == len(mapping)``) treated that
    fold as non-canonical, sending every session with an in-scope
    person carrying even one PII attribute (phone, email, city, …) down
    the non-canonical branch at every leak-detected call site
    (``extractor.py:1031``, ``:2147``) — repair never ran; cloud egress
    silently blocked. It was also a no-op for the collision it claimed
    to catch: two distinct entity NAMES colliding onto one placeholder
    is already resolved by ``reverse.setdefault`` keeping only the
    first (``placeholders.py:870``), so there is no duplicate left for
    this function to find. Enforcing injectivity belongs at the
    construction site (the LLM-hint merge), not here.

    Empty mapping is canonical (no entries to validate).
    """
    if not mapping:
        return True
    if placeholder_side == "key":
        placeholder_entries = mapping.keys()
        other_entries = mapping.values()
    else:
        placeholder_entries = mapping.values()
        other_entries = mapping.keys()
    if not all(PLACEHOLDER_SHAPE_RE.match(str(v)) for v in placeholder_entries):
        return False
    return not any(PLACEHOLDER_SHAPE_RE.match(str(k)) for k in other_entries)


# ---------------------------------------------------------------------------
# Resolution map — core + secondary (SOTA) merge. CORE-LAST precedence and
# the `observed is None` CORE-UNSCOPED sentinel are load-bearing invariants.
# ---------------------------------------------------------------------------


def _resolution_map(
    reverse: dict[str, str],
    sota_bindings: dict[str, str],
    observed: set[str] | None,
) -> dict[str, str]:
    """The ONE legality/resolution map — built once, consumed by both
    :func:`_check_mapping_totality` (as its legality domain) and
    :func:`_apply_bindings` (as its substitution map).

    ``observed`` is ``None`` -> **CORE UNSCOPED** (today's behaviour;
    every non-SOTA deanon path, and the ``_skip_sota`` path where SOTA
    never ran): every ``reverse`` entry is legal, ``sota_bindings`` is
    merged in underneath it. An empty-set default here instead of ``None``
    would scope CORE to nothing and drop every fact on those paths — see
    the callers' ``observed: set[str] | None = None`` declarations.

    ``observed`` is a ``set`` -> **CORE SCOPED** to it: a ``reverse``
    entry resolves only when SOTA was actually shown its placeholder
    (``key in observed`` — a token in the rendered facts SOTA saw, or in
    the anonymized transcript). Every ``sota_bindings`` entry whose key
    is NOT in ``observed`` is SOTA's own mint and resolves too. A key
    present in both domains is a CONFLICT, caught upstream by
    :func:`_check_mapping_totality` before an accepted delta ever reaches
    this scoped branch — the map still reflects the tie-break if asked to
    resolve one directly (unit-tested, T2f).

    **CORE PRECEDENCE (named invariant, T2f) — CORE-LAST BY CONSTRUCTION.**
    In both branches ``reverse`` entries are applied AFTER
    ``sota_bindings``, so any key present in both resolves to CORE's
    value. SOTA can never override or misresolve against the core map.
    This is deliberate construction order, not an accident of dict-spread
    — do not reorder the two ``.update()`` calls below.
    """
    resolved: dict[str, str] = {}
    if observed is None:
        resolved.update(
            (k, v)
            for k, v in sota_bindings.items()
            if isinstance(k, str) and isinstance(v, str) and k and v
        )
        resolved.update(
            (k, v)
            for k, v in reverse.items()
            if isinstance(k, str) and isinstance(v, str) and k and v
        )
    else:
        resolved.update(
            (k, v)
            for k, v in sota_bindings.items()
            if isinstance(k, str) and isinstance(v, str) and k and v and k not in observed
        )
        resolved.update(
            (k, v)
            for k, v in reverse.items()
            if isinstance(k, str) and isinstance(v, str) and k and v and k in observed
        )
    return resolved


def _check_mapping_totality(
    graph: SessionGraph,
    anon_facts: list[dict],
    reverse_mapping: dict,
    *,
    sota_bindings: dict | None = None,
    observed: set[str] | None = None,
    diagnostic_key: str = "totality_orphans",
    stage: str = "anonymizer",
) -> list[str]:
    """Diagnostic check: every placeholder in any anonymized fact must
    resolve against :func:`_resolution_map` (``reverse_mapping`` plus
    ``sota_bindings``, scoped to ``observed`` — see that function) — the
    SAME legality domain :func:`_apply_bindings` substitutes with at
    deanon time.  Checked against the reverse map — not the forward map —
    because a placeholder present only in the forward map's values still
    fails to translate at deanon time.  Surfaces violations to ``logger``
    and ``graph.diagnostics[diagnostic_key]`` so prompt regressions are
    visible rather than silently shedding facts.

    Returns the sorted list of offending tokens (orphans, plus conflicts
    when ``observed`` is given — see below); ``[]`` when the mapping is
    total.  Two exits are intentionally explicit rather than falling
    through to an implicit ``None`` — the empty-``anon_facts`` early
    return and the empty-orphans path both ``return []`` so callers can
    do a plain truthiness test on the result without a
    falsy-``None``-vs-falsy-``[]`` trap.

    Generalised to cover two call sites: the pre-SOTA anonymizer check
    (``sota_bindings=None``, ``observed=None``, default
    ``diagnostic_key``/``stage``) and the post-SOTA check
    (``sota_bindings=sota_bindings, observed=<rendered tokens>,
    diagnostic_key="sota_pending_orphans", stage="sota_enrichment"``).
    For the anonymizer stage the caller only logs the violation — the
    affected fact is dropped by :func:`_apply_bindings`'s residual sweep
    (the single deanon exit gate), the correct fail-closed semantic
    there.  For the SOTA-enrichment
    stage the caller (:func:`~paramem.graph.extractor._sota_pipeline`)
    instead REJECTS THE WHOLE DELTA on a non-empty verdict and falls back
    to the pre-enrichment local-extract facts — a single unbound
    placeholder no longer sheds just that one fact.  Matches both braced
    (``{Event_1}``) and bare (``Event_1``) placeholder forms via
    :data:`PLACEHOLDER_TOKEN_RE` — harmless for the pre-SOTA call, whose
    facts only ever carry bare tokens.

    When ``sota_bindings`` is given and ``observed`` is a set, any
    ``sota_bindings`` KEY that is also in ``observed`` is a CONFLICT
    (SOTA referencing/rebinding something it was already shown as a core
    reference) and is folded into the returned verdict.  When
    ``observed`` is ``None`` (CORE unscoped — today's behaviour), the
    conflict scan instead flags any KEY present in both ``sota_bindings``
    and ``reverse_mapping`` with a DIFFERING value — informational only,
    NOT folded into the verdict, since :func:`_resolution_map`'s
    CORE-wins tie-break already resolves it safely.  Both are recorded
    (sorted) to ``graph.diagnostics["sota_binding_collisions"]``.

    Does not mutate inputs and does not change the data flow.
    """
    orphans: set[str] = set()
    if sota_bindings:
        if observed is not None:
            collisions = sorted(k for k in sota_bindings if k in observed)
        else:
            collisions = sorted(
                k
                for k, v in sota_bindings.items()
                if k in reverse_mapping and reverse_mapping[k] != v
            )
        if collisions:
            logger.warning(
                "SOTA binding collision: %d placeholder(s) present in both "
                "sota_bindings and reverse_mapping with differing values "
                "(reverse_mapping wins): %s.",
                len(collisions),
                collisions[:5],
            )
            graph.diagnostics["sota_binding_collisions"] = collisions
            if observed is not None:
                orphans |= set(collisions)
    if not anon_facts:
        return sorted(orphans) if orphans else []
    resolvable = set(_resolution_map(reverse_mapping, sota_bindings or {}, observed))
    for f in anon_facts:
        if not isinstance(f, dict):
            continue
        for field in ("subject", "object"):
            for t in PLACEHOLDER_TOKEN_RE.findall(str(f.get(field, ""))):
                name = t[0] or t[1]
                if name not in resolvable:
                    orphans.add(name)
    # Q3 completeness: a binding value may itself contain a bare placeholder
    # (e.g. "Senior Engineer at Org_9") that never resolves — the braced
    # pass exposes it but nothing substitutes it, so it drops silently.
    for value in (sota_bindings or {}).values():
        if not isinstance(value, str):
            continue
        for t in PLACEHOLDER_TOKEN_RE.findall(value):
            name = t[0] or t[1]
            if name not in resolvable:
                orphans.add(name)
    if orphans:
        ordered = sorted(orphans)
        if stage == "sota_enrichment":
            logger.warning(
                "SOTA enrichment binding-totality violation: %d orphan/conflict "
                "placeholder(s) in anon_facts not resolvable: %s. The enrichment "
                "delta will be rejected; local-extract facts are kept instead.",
                len(ordered),
                ordered[:5],
            )
        else:
            logger.warning(
                "Anonymization mapping-totality violation: %d orphan placeholder(s) "
                "in anon_facts not resolvable: %s. Affected fact(s) will be "
                "dropped at the residual placeholder sweep post-deanon.",
                len(ordered),
                ordered[:5],
            )
        graph.diagnostics[diagnostic_key] = ordered
        return ordered
    return []


# ---------------------------------------------------------------------------
# Detection — declared-vocabulary scan (substring, no \b, no regex) plus the
# shape regex above as the fail-closed net.
# ---------------------------------------------------------------------------


def _declared_placeholder_tokens(
    reverse: dict[str, str], sota_bindings: dict[str, str] | None = None
) -> set[str]:
    """The declared placeholder-token vocabulary for a deanon call.

    Every key in ``reverse`` (the anonymizer's CORE map, ``placeholder ->
    entity_name``) plus every key in ``sota_bindings`` (SOTA's own minted
    placeholders) is a token this session's pipeline actually declared —
    the ONE vocabulary the fail-closed predicate invariant and residual
    sweep (:func:`_apply_bindings`) test membership against.

    Deliberately NOT :data:`PLACEHOLDER_TOKEN_RE`: that pattern's ``\\b``
    anchor misses a token glued onto a longer identifier
    (``language_proficiency_Language_3`` does not match
    ``\\bLanguage_3\\b``) — exactly the class of bug this
    vocabulary-based check exists to catch. Token SHAPE is irrelevant
    here (bare today, braced after a future format flip); only
    DECLARED-ness — membership in one of the two mapping tables —
    matters, so this helper survives that flip unchanged.
    """
    tokens: set[str] = {k for k in reverse if isinstance(k, str) and k}
    if sota_bindings:
        tokens.update(k for k in sota_bindings if isinstance(k, str) and k)
    return tokens


def _contains_declared_token(text: str, declared: set[str]) -> bool:
    """True iff ``text`` contains any token in ``declared`` as a literal
    substring, anywhere in the string — no regex, no word-boundary
    anchor. This is what lets the check catch a token glued into a
    longer identifier (``language_proficiency_Language_3``) that
    :data:`PLACEHOLDER_TOKEN_RE` misses.
    """
    return any(tok in text for tok in declared)


# ---------------------------------------------------------------------------
# Table construction — from entity records.
# ---------------------------------------------------------------------------

# PII-bearing attribute keys.  The local extractor projects these onto
# ``Entity.attributes`` rather than as standalone entities or relations
# — their values appear verbatim in the source transcript and must be
# scrubbed before the anonymized transcript is sent to a SOTA cloud
# provider.  Consumed by :func:`_build_anonymization_mapping`.
#
# The list is intentionally conservative: keys with a high prior of
# carrying personally-identifying or location data.  Operators can
# extend by adding entries here.
_PII_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
        "last_name",
        "first_name",
        "email",
        "phone",
        "linkedin",
        "city",
        "country",
        "location",
    }
)

# Primitive-layer default scope for :func:`_build_anonymization_mapping`
# and (via re-export) ``check_anonymization_leaks`` /
# ``extract_pii_names_with_ner`` when no explicit ``pii_scope`` is
# passed.  Preserves the historical hardcoded ``{person, place}`` scope
# of these primitives so consolidation (``_sota_pipeline``) and any
# direct callers keep the prior leak-detection coverage.
#
# This is *not* the cloud-egress policy default — that is
# ``_CLOUD_EGRESS_DEFAULT_SCOPE`` in :mod:`paramem.graph.extractor`,
# narrower and configurable via ``SanitizationConfig.cloud_scope`` —
# different concern, different default.  The primitive has no policy
# opinion; the helper does.
_DEFAULT_PII_SCOPE: frozenset[str] = frozenset({"person", "place"})


def _build_anonymization_mapping(
    entities: Iterable[Entity],
    llm_mapping: dict,
    *,
    pii_scope: set[str] | frozenset[str] | None,
    speaker_name: str | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Single source of truth for the real → placeholder mapping.

    The anonymizer LLM is good at producing **anon_facts** (canonical
    predicates, compound-fact splits, dropping unsafe relations) but
    historically untrustworthy at producing a **complete mapping**:
    it routinely omits the bare first-name form when it has the full
    name, never sees ``Entity.attributes`` at all (so PII embedded in
    attributes leaks), and emits ambiguous pairs that have to be
    canonicalized after the fact.  Three accreting "Bug X fix" helpers
    (speaker-name seeding, attribute-extend, ambiguous-pair drop) used
    to patch the LLM's output incrementally.

    This builder replaces that pattern with one deterministic walk:

    1. **Mint placeholders for in-scope entity names — except the
       speaker.**  The graph knows the canonical entity inventory; we
       don't need the LLM to enumerate it.  Placeholder allocation is
       per-prefix (``Person_1, Person_2, …, Org_1, …``) via
       :func:`mint_placeholder`.  The speaker entity
       (``entity.speaker_id is not None``) is EXCLUDED from this mint:
       ``speaker{N}`` is already an anonymized handle (CLAUDE.md's
       "ONE lowercase ``speaker{N}`` everywhere") — minting a
       ``Person_N`` for it would anonymize something already anonymous
       and draw it into the session-dependent, positionally-allocated
       ``Person_N`` counter.  When the entity's own ``name`` is a
       disclosed display form (differs from ``entity.speaker_id``), that
       name folds onto ``entity.speaker_id`` in the forward map instead
       of onto a fresh placeholder — still scrubbed, just onto the
       anchor.  ``reverse`` never gets an entry for the anchor: display
       names are resolved at the fact-render boundary
       (:func:`~paramem.server.speaker.resolve_speaker_name`), never
       baked into graph edges.
    2. **Add PII attribute values under the parent entity's
       placeholder** (the speaker's own ``speaker_id`` counts as its
       "placeholder" here).  Reusing the placeholder is privacy-correct
       (SOTA only needs tokens scrubbed, not unique placeholders per
       attribute) and keeps the mapping canonical (no novel
       placeholder shapes).  These attribute values land in the
       *forward* map only — the reverse map records only the entity
       name for each placeholder, so deanonymization can never
       restore an attribute value (phone, email, …) where the entity
       name belongs.
    3. **Speaker-name seeding** (legacy Bug A).  When the runtime
       knows the speaker's display name and that name isn't already
       in the mapping (e.g. anonymous-id session, full-name vs
       first-name mismatch), reuse the speaker entity's anchor
       (``entity.speaker_id``) or — if no speaker entity is in scope —
       fall back to LLM's exact or full-name match (``"Alex"`` or
       ``"Alex Rivera"`` → reuse ``Person_1``) or mint a fresh
       ``Person_N``.  The no-speaker-entity fallback is unchanged by
       the anchor (R7, out of scope): it can still mint a ``Person_N``
       for the speaker's display name in the rare case the graph has no
       speaker entity at all.
    4. **Preserve LLM hints — except any keyed on or valued at the
       speaker.**  Entries the LLM emitted that the deterministic build
       does not cover (typically relation participants the graph
       doesn't know about — e.g. ``Honda`` mentioned in a relation but
       not a graph entity) are merged in.  When the LLM's entry
       conflicts with a deterministic one, deterministic wins (we trust
       the graph).  A key that is speaker-id-shaped
       (:func:`~paramem.graph.name_match.is_speaker_id`) — e.g. a
       hallucinated ``{"speaker0": "Person_1"}`` — is dropped outright
       (forward *and* reverse), since treating the anchor itself as a
       "real name" to be re-mapped would corrupt it in both directions.
       A *value* that is speaker-id-shaped — e.g. ``{"RealName":
       "speaker0"}``, an anonymizer hallucination scrubbing a real name
       onto the anchor — keeps the forward scrub (harmless, and the
       only thing standing between that real name and the cloud) but
       drops the reverse write: a reverse entry keyed on ``speaker0``
       would restore that real name onto every speaker-subject fact.

    The companion :func:`_check_mapping_totality` runs after this
    builder as a diagnostic for the orthogonal concern of LLM
    placeholders that surface in ``anon_facts`` without any mapping
    entry — those facts are dropped by the residual placeholder sweep
    post-deanon, with the violation logged for monitoring.

    Args:
        entities: The entity records to walk — ``graph.entities`` for the
            session tier; for the graph tier (cumulative, post-merge
            graph, which carries no entity types of its own) one
            :class:`~paramem.graph.schema.Entity` per real name found by
            the chunk's local-model anonymization pass, typed via
            :func:`prefix_to_entity_type` on the placeholder that pass
            minted (see :func:`~paramem.training.consolidation.
            ConsolidationLoop._run_graph_enrichment`). This is the sole
            input the function reads names/types/attributes from — it has
            no other opinion about where entities come from.
        llm_mapping: Canonicalised ``{real_name: placeholder}`` mapping
            from :func:`_normalize_anonymization_mapping`.  Treated as
            a hint, not truth.
        pii_scope: Entity-types in scope for anonymization (e.g.
            ``{"person", "place"}``).  Out-of-scope entities pass
            through unscrubbed by design.
        speaker_name: Runtime-known display name of the session's
            speaker.  When set, this name is guaranteed to be covered.

    Returns:
        ``(forward, reverse)`` — ``forward`` is the many-to-one
        ``{real_name | attr_value: placeholder}`` mapping that feeds
        ``_anonymize_transcript`` and ``check_anonymization_leaks``
        (both in :mod:`paramem.graph.extractor`).  ``reverse`` is the
        one-to-one ``{placeholder: entity_name}`` map consumed by the
        deanon path (:func:`deanonymize_text`, :func:`_apply_bindings`)
        — attribute values are deliberately absent so a folded
        placeholder always restores to the entity name.  For the
        speaker, "placeholder" in ``forward`` is the anchor
        (``entity.speaker_id``, e.g. ``"speaker0"``) rather than a
        minted value, and ``reverse`` carries NO entry for it at all —
        the anchor is never a mint target and display names are never
        restored into graph edges.
    """
    scope: frozenset[str] = _DEFAULT_PII_SCOPE if pii_scope is None else frozenset(pii_scope)
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}

    # Mint placeholders for in-scope entities and fold in their PII
    # attribute values under the same placeholder.
    speaker_entity_placeholder: str | None = None
    for entity in entities:
        if entity.entity_type not in scope:
            continue
        if not entity.name:
            continue
        if entity.speaker_id is not None:
            # The speaker anchor — never mint a Person_N for it (see the
            # module-level docstring, point 1).  entity.speaker_id itself
            # (e.g. "speaker0") IS the anchor; entity.name may already be
            # that same anonymous form, or a disclosed display name.
            if speaker_entity_placeholder is None:
                speaker_entity_placeholder = entity.speaker_id
            if entity.name != entity.speaker_id:
                mapping.setdefault(entity.name, entity.speaker_id)
            # No reverse entry: the anchor is never a mint target, and
            # display names are never restored into graph edges.
            attrs = entity.attributes or {}
            for attr_key, attr_value in attrs.items():
                if attr_key not in _PII_ATTRIBUTE_KEYS:
                    continue
                if not isinstance(attr_value, str) or not attr_value.strip():
                    continue
                mapping.setdefault(attr_value, entity.speaker_id)
            continue
        prefix = entity_type_to_prefix(entity.entity_type)
        if entity.name not in mapping:
            # Mint against the union of what is already allocated in
            # `mapping` AND `llm_mapping` — a token the LLM already used
            # for a DIFFERENT real name (merged into `mapping` only
            # later, in the loop below) is otherwise invisible to the
            # mint and two distinct real names can collide onto the
            # same placeholder (FIX 1).  An `llm_mapping` entry keyed on
            # THIS entity's own name is excluded from the scan — that is
            # the LLM's (possibly disagreeing) hint for the SAME real
            # name, not a competing claimant, and letting it block this
            # mint would spuriously inflate the placeholder count and
            # duplicate the reverse map every time the LLM hint happens
            # to agree with the graph.
            other_llm_values = (v for k, v in llm_mapping.items() if k != entity.name)
            mapping[entity.name] = mint_placeholder([*mapping.values(), *other_llm_values], prefix)
        placeholder = mapping[entity.name]
        # Only entity names enter the reverse map.  ``setdefault`` so the
        # first-seen entity wins if two distinct entities collide on the
        # same placeholder via an LLM hint downstream.
        reverse.setdefault(placeholder, entity.name)
        attrs = entity.attributes or {}
        for attr_key, attr_value in attrs.items():
            if attr_key not in _PII_ATTRIBUTE_KEYS:
                continue
            if not isinstance(attr_value, str) or not attr_value.strip():
                continue
            mapping.setdefault(attr_value, placeholder)
            # NOTE: deliberately not adding to ``reverse``.  Folding
            # attribute values into the forward map is privacy-correct
            # (one placeholder scrubs every PII surface form) but the
            # reverse direction must restore the entity name, not an
            # attribute value.

    # Speaker-name seeding: ensure the runtime-known speaker name is covered.
    if speaker_name and speaker_name not in mapping:
        if speaker_entity_placeholder is not None:
            # Speaker is among the walked entities.  ``speaker_entity_placeholder``
            # is the anchor (entity.speaker_id, e.g. "speaker0"), never a
            # minted Person_N — reuse it so the runtime-known display
            # name folds onto the same anonymous handle as the entity's
            # own name.  No reverse entry: the anchor never restores a
            # display name into graph edges.
            mapping[speaker_name] = speaker_entity_placeholder
        else:
            # No speaker entity in scope — fall back to LLM hints
            # (exact or full-name match) or mint a fresh Person_N.
            speaker_lower = speaker_name.lower()
            reused: str | None = None
            for key, placeholder in llm_mapping.items():
                if not isinstance(key, str):
                    continue
                key_lower = key.lower()
                if key_lower == speaker_lower or key_lower.startswith(speaker_lower + " "):
                    reused = placeholder
                    break
            if reused is not None:
                mapping[speaker_name] = reused
                reverse.setdefault(reused, speaker_name)
            else:
                person_prefix = entity_type_to_prefix("person")
                # Allocate Person_N taking BOTH the CORE mapping and
                # llm_mapping values into account so we don't collide
                # with an LLM-emitted Person_K.
                merged_for_idx = dict(mapping)
                for k, v in llm_mapping.items():
                    merged_for_idx.setdefault(k, v)
                fresh = mint_placeholder(merged_for_idx.values(), person_prefix)
                mapping[speaker_name] = fresh
                reverse.setdefault(fresh, speaker_name)

    # Merge in LLM hints not already covered (typically relation-participant
    # placeholders for entities absent from ``entities``).  Deterministic
    # entries win on conflict; LLM-only entries are added.  LLM-emitted
    # keys are entity-name-shaped (the anonymizer prompt operates on
    # relation participants, not attributes), so they are safe to enter
    # the reverse map.  The reverse write is NOT gated on the forward
    # write winning — anon_facts emitted under the LLM's (losing)
    # placeholder still need a reverse entry to deanonymize.
    #
    # A key that is speaker-id-shaped (``is_speaker_id``) — e.g. a
    # hallucinated ``{"speaker0": "Person_1"}`` — is dropped outright
    # (forward and reverse): the anchor itself is never a "real name"
    # to be re-mapped.  A value that is speaker-id-shaped — e.g.
    # ``{"RealName": "speaker0"}``, the LLM scrubbing a real name onto
    # the anchor — keeps the forward scrub (still useful: it is what
    # keeps that real name out of ``anon_transcript``) but drops ONLY
    # the reverse write, which would otherwise restore the real name
    # onto every ``speaker0``-subject fact at deanon.
    for k, v in llm_mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if is_speaker_id(k):
            continue
        mapping.setdefault(k, v)
        if is_speaker_id(v):
            continue
        reverse.setdefault(v, k)

    return mapping, reverse


# ---------------------------------------------------------------------------
# Deanonymization — the exit gate for facts, and the free-text deanon.
# ---------------------------------------------------------------------------

# The fields of a fact dict that constitute a `Relation` — exactly the
# keys read at the `Relation(**fact)`-equivalent construction site in
# `_sota_pipeline` (subject/predicate/object/relation_type/confidence/
# symmetric; `speaker_id` is stamped separately from the session, never
# read off the fact). Any OTHER key on a fact dict (e.g. an `evidence`
# field an LLM invents) never reaches `Relation` and therefore cannot
# leak a placeholder anywhere observable — the residual sweep in
# `_apply_bindings` only tests these fields, and the SOTA enrichment
# delta boundary (`_parse_enrichment_delta`) strips any other key from
# `add`/`modify` entries before they ever enter `enriched_anon`.
_FACT_FIELDS: frozenset[str] = frozenset(
    {"subject", "predicate", "object", "relation_type", "confidence", "symmetric"}
)


def _apply_bindings(
    facts: list[dict],
    reverse: dict[str, str],
    sota_bindings: dict[str, str],
    observed: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """De-anonymize facts via state-machine substitution — the SINGLE
    deanon exit gate, in three ordered steps:

    1. **Predicate invariant (BEFORE substitution).** A fact whose
       ``predicate`` field contains ANY token from
       :func:`_declared_placeholder_tokens` (``reverse`` keys union
       ``sota_bindings`` keys), as a literal substring, is dropped
       outright — no splitting, no repair. This runs first so a
       poisoned predicate (``at_Org_1``) is never "resolved" into a
       garbage predicate (``at_Acme``): the predicate field is never a
       substitution target below, so checking it after substitution
       would find nothing wrong with an already-corrupted predicate.
    2. **Substitute** subject/object with :func:`_resolution_map`
       (``reverse``, ``sota_bindings``, ``observed``) — the SAME
       legality domain :func:`_check_mapping_totality` checks, rendered
       in both braced and bare form:

       * **Anonymizer reverse map** (``reverse`` arg) —
         ``placeholder -> entity_name`` produced by
         :func:`_build_anonymization_mapping`.  Earlier revisions
         inverted the forward mapping here, which was lossy when PII
         attributes folded onto the entity placeholder; the explicit
         reverse is now produced alongside the forward map.
       * **SOTA bindings** (``sota_bindings`` arg) —
         ``placeholder_name -> real_text`` that SOTA emitted alongside
         its enriched facts (new entities SOTA minted, e.g.
         ``Event_1``).
       * **``observed``** (trailing, defaulted ``None``) — ``None``
         means CORE UNSCOPED (every ``reverse`` entry is legal; this is
         what makes the five pre-existing positional-only call sites in
         ``test_extraction_pipeline.py`` keep passing unchanged as the
         CORE-unscoped regression net).  A ``set`` means CORE SCOPED to
         it — see :func:`_resolution_map`.  ``reverse`` wins on any key
         collision in EITHER mode (CORE PRECEDENCE, T2f) — deterministic
         entity names over SOTA-sourced values, never the reverse.

       The union round-trips a placeholder regardless of which form
       (braced or bare) it was actually emitted in: SOTA's contract asks
       for braced minted placeholders and bare anonymizer placeholders,
       but models don't always comply, so both maps are tried against
       both forms. Braced literal substitution runs first (unambiguous,
       no word-boundary needed), then word-boundary substitution over
       the same map catches bare occurrences (``Person_2's cousin`` ->
       ``Alex's cousin``) and resolves any bare placeholder nested
       inside a bound value (``"Senior Engineer at Org_1"`` ->
       ``"Senior Engineer at Acme"``).
    3. **Residual sweep, any FACT field (AFTER substitution).** Any
       field in :data:`_FACT_FIELDS` (the fields that actually reach
       ``Relation`` — an LLM-invented extra like ``evidence`` is never
       swept, since it never reaches the graph either) still containing
       a declared token (:func:`_contains_declared_token`) — or a
       placeholder-shaped token per :data:`PLACEHOLDER_TOKEN_RE`, kept
       as the fail-closed backstop for an UNDECLARED orphan the
       predicate/declared-token checks cannot see — is dropped. Causes,
       direct-call context (this function invoked in isolation, e.g. by
       its own unit tests):
         a. SOTA introduced a braced placeholder but omitted its binding.
         b. SOTA emitted a bare placeholder that was never in the
            anonymizer mapping (anonymizer leak).
         c. Composite strings where one of multiple placeholders
            couldn't be resolved.
       Inside the full pipeline (:func:`~paramem.graph.extractor._sota_pipeline`),
       causes (a) and (b) are now intercepted upstream by
       :func:`_check_mapping_totality`'s SOTA-enrichment-stage rejection
       gate — a bad mint rejects the WHOLE delta before it reaches this
       function, rather than shedding just the one fact.  This sweep
       remains the fail-closed backstop for cause (c) and for a genuine
       anonymizer-stage leak (R3).

    Non-dict entries in ``facts`` are silently skipped — never counted
    in any returned list.

    Returns ``(kept_facts, predicate_dropped, residual_dropped)`` — the
    two drop categories are returned ALREADY partitioned (callers must
    not recompute the split): ``predicate_dropped`` holds the exact
    pre-substitution input dict for each fact step 1 removed;
    ``residual_dropped`` holds the post-substitution copy for each fact
    step 3 removed.

    Replaces the previous LLM-based deanon attempt that crashed on the
    largest chunk's prompt with ``device not ready`` (VRAM exhaustion on
    Mistral 7B at 8 GiB). Also replaces the regex-based binding recovery
    (``_extract_sota_bindings``) which produced bogus mappings under
    multi-token replace blocks (bug 5).
    """
    declared = _declared_placeholder_tokens(reverse, sota_bindings)

    # Step 1 — predicate invariant, BEFORE substitution.
    pre_filtered: list[dict] = []
    predicate_dropped: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        if _contains_declared_token(str(f.get("predicate", "")), declared):
            predicate_dropped.append(f)
            continue
        pre_filtered.append(f)

    # Step 2 — substitute subject/object (unchanged semantics).
    resolve = _resolution_map(reverse, sota_bindings, observed)
    braced_map: dict[str, str] = {braced(k): v for k, v in resolve.items()}

    substituted: list[dict] = []
    for f in pre_filtered:
        subj = str(f.get("subject", ""))
        obj = str(f.get("object", ""))
        # Pass 1: braced literal substring replace (unambiguous, no
        # word-boundary needed).
        for braced_token, real in braced_map.items():
            if braced_token in subj:
                subj = subj.replace(braced_token, real)
            if braced_token in obj:
                obj = obj.replace(braced_token, real)
        # Pass 2: bare word-boundary substitution over the SAME union map
        # (apostrophes / surrounding punctuation handled; also resolves
        # any bare token exposed by pass 1, e.g. nested-value binding).
        subj = _substitute_whole_words(subj, resolve)
        obj = _substitute_whole_words(obj, resolve)
        substituted.append({**f, "subject": subj, "object": obj})

    # Step 3 — residual sweep, ANY FACT field (never a non-fact field an
    # LLM invented — see `_FACT_FIELDS`), fail-closed. A fact is "clean"
    # only if none of its fact fields carries a declared token or a
    # placeholder-shaped token; either is grounds to drop the whole
    # fact.
    kept: list[dict] = []
    residual_dropped: list[dict] = []
    for f in substituted:
        residual = any(
            isinstance(v, str)
            and (PLACEHOLDER_TOKEN_RE.search(v) or _contains_declared_token(v, declared))
            for v in (f.get(field) for field in _FACT_FIELDS)
        )
        if residual:
            residual_dropped.append(f)
        else:
            kept.append(f)

    return kept, predicate_dropped, residual_dropped


def deanonymize_text(text: str, reverse: dict[str, str]) -> str:
    """Restore real names in cloud-returned text via the reverse mapping.

    ``reverse`` is the ``{placeholder: real_name}`` map produced
    alongside the forward map by :func:`_build_anonymization_mapping`
    (or by the dict-inversion in
    :func:`~paramem.graph.extractor.extract_and_anonymize_for_cloud` for
    LLM-emitted mappings, which are one-to-one by contract).  This
    function never inverts a forward map itself — that inversion was
    lossy when the forward map folded PII attribute values onto the
    entity placeholder, and the asymmetry now lives in the producer.

    Word-boundary anchored, so a placeholder embedded in unrelated
    text doesn't match.  Idempotent on text without placeholders or
    with an empty reverse map.
    """
    if not text or not reverse:
        return text
    return _substitute_whole_words(text, reverse)
