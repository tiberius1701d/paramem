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
from paramem.graph.schema import Relation, SessionGraph
from paramem.graph.schema_config import anonymizer_prefix_to_type, anonymizer_type_to_prefix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shape — the ONE placeholder-shape source, serving both the anchored
# full-string validator and the unanchored in-text detector.
# ---------------------------------------------------------------------------

# One-or-more PascalCase segments, underscore-joined, + "_" + positive
# integer. The prefix vocabulary is open: "Person" / "City" / "Org" /
# "Thing" are common (configured in configs/schema.yaml) but any
# PascalCase noun is well-formed — a model is free to mint
# "University_1" / "Project_1" / "Language_1" for a type none of the
# common prefixes fit. The prefix itself may be MULTI-SEGMENT
# (underscore-joined PascalCase words, e.g. "Home_Address_1",
# "Car_Plate_1") — a model emits these for in-scope categories whose
# natural label is itself multi-word. Only the FINAL "_\d+" is the
# mandatory numeric suffix; "Foo_Bar" (no trailing digits) does not
# match.
_BARE_PLACEHOLDER_SHAPE = r"[A-Z][A-Za-z]*(?:_[A-Z][A-Za-z]*)*_\d+"

# Anchored full-string validator — used by table normalize/validate
# (:func:`_normalize_anonymization_mapping`) to classify whether a table
# ENTRY (not embedded in surrounding text) is placeholder-shaped.
# Collapses the previous three-way split between
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

    THE only place a placeholder token string is built. Every mint in
    this module — including :func:`_build_anonymization_mapping`'s
    per-prefix and speaker-name-fallback mints — goes through this one
    function; never re-implement a local counter or scanning closure
    at a call site.
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

    EDGE-AWARE boundaries: a match at ``pos`` requires a boundary on a
    side only if the KEY's edge char on that side is a word char
    (:func:`_is_word_char`).  A key whose first/last char is a
    non-word char (e.g. ``"+49 151 2345"``) needs no boundary check on
    that side and can therefore match starting or ending mid-run —
    fixing the historical bug where a key starting with a non-word char
    was never attempted because matches were only tried at word-char
    positions.  A key that IS word-char-bounded on a side (``"Bill"``)
    still requires a non-word (or string-edge) neighbour there, so
    ``"Bill"`` matches standalone but never inside ``"Billing"``.

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
    sees the mapping — see
    :func:`~paramem.training.graph_enrich.run_graph_enrichment`.
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
        matched = False
        for key in keys_sorted:
            klen = len(key)
            end = pos + klen
            if end > n:
                continue
            if _is_word_char(key[0]) and pos > 0 and _is_word_char(text[pos - 1]):
                continue
            if _is_word_char(key[-1]) and end < n and _is_word_char(text[end]):
                continue
            if text[pos:end] != key:
                continue
            replacement = normalized[key]
            if not isinstance(replacement, str):
                continue
            parts.append(replacement)
            pos = end
            matched = True
            break
        if not matched:
            parts.append(text[pos])
            pos += 1
    return "".join(parts)


def _build_anon_facts(relations: Iterable[Relation], mapping: dict[str, str]) -> list[dict]:
    """Build the anonymized fact array from ``relations`` and the forward
    ``mapping`` — THE single construction that feeds off the
    anonymizer's fact-free contract: the anonymizer LLM returns the
    ``mapping`` and its own ``anonymized_transcript`` rewrite, but never
    a fact array, so every anonymizer-facing fact array is built HERE,
    mechanically, from ``relations``.

    One dict per relation: ``subject``/``object`` are substituted through
    ``mapping`` via :func:`_substitute_whole_words`; ``predicate``,
    ``relation_type``, and ``confidence`` are copied VERBATIM — the
    predicate is NEVER a substitution target, so a placeholder cannot be
    glued into it at this stage (the motivating bug,
    ``language_proficiency_Language_3``, can still occur in SOTA's
    *returned* facts, which is why the deanon-stage predicate invariant
    in :func:`_apply_bindings` stays).  The output count is exactly
    ``len(relations)`` by construction — a fact can never be lost,
    reworded, or dropped by the anonymizer, because the anonymizer never
    returns one.

    The SOLE caller is :func:`~paramem.graph.cloud_egress.anonymize_for_cloud`
    (A) step 8 — every one of the five migrated paths (session-tier
    extraction, graph-tier enrichment, chat egress, and their calibration
    harnesses) reaches this function exclusively through that one chain,
    never directly.  ``_graph_enrich_with_sota`` does NOT call this
    function — its anonymized triples come from
    ``payload.anon_facts`` (already built by (A)), zipped positionally
    onto the chunk's ``triples`` (see that function's docstring).
    ``mapping`` must already be the COMPLETE forward map (i.e.
    :func:`_build_anonymization_mapping`'s output) — a partial/HINT-only
    map here reproduces the exact leak this function exists to prevent.
    """
    return [
        {
            "subject": _substitute_whole_words(r.subject, mapping),
            "predicate": r.predicate,
            "object": _substitute_whole_words(r.object, mapping),
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in relations
    ]


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
    corrupting every live consumer of the derived type: SOTA-minted-entity
    type inference in :func:`~paramem.graph.extractor._sota_pipeline`
    (``:2660``) and entity-surface correction in
    :mod:`paramem.graph.entity_correction` (``:276``). Stripping braces
    first means this function survives that flip unchanged.
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

    THE CORE table (``placeholder_side="value"``, the default) has
    exactly ONE caller —
    :func:`~paramem.graph.cloud_egress.anonymize_for_cloud`'s step 4 —
    and that caller's ``stats`` is a LIVE signal: it flows into
    ``AnonymizedPayload.norm_stats`` and, for the session tier, into
    ``graph.diagnostics["mapping_ambiguous_dropped"]``.  Before the
    anon/deanon unification, every migrated caller ran a SECOND,
    redundant normalize on an already-canonical table (the internal call
    that is now this function's ONE call site had already dropped every
    ambiguous pair), so the outer ``stats["dropped"]`` could only ever be
    ``0`` — a structurally-dead diagnostic.  The ``bindings`` table
    variant (``placeholder_side="key"``) still has more than one caller
    (:meth:`~paramem.graph.cloud_egress.CloudScope.for_response`, and the
    unrelated legacy per-session delta parser
    :func:`~paramem.graph.extractor._parse_enrichment_delta`) and its
    ``stats`` is not currently surfaced to a diagnostic by either.
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
    every deanon call that has no SOTA-observed scope to pass — e.g. a
    call outside the SOTA-enrichment cycle, or a unit test exercising
    :func:`_apply_bindings`/:func:`_resolution_map` directly): every
    ``reverse`` entry is legal, ``sota_bindings`` is merged in
    underneath it. An empty-set default here instead of ``None``
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
    resolve one directly (unit-tested).

    **CORE PRECEDENCE (named invariant) — CORE-LAST BY CONSTRUCTION.**
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
    total.  ONE exit point for a non-empty verdict — whether the orphans
    came from the collision scan (``sota_bindings`` vs. ``observed``/
    ``reverse_mapping``) or the per-fact placeholder scan — so
    ``graph.diagnostics[diagnostic_key]`` is written on every non-empty
    verdict, never silently dropped because ``anon_facts`` happened to be
    empty (a real regression this function's caller-side counter,
    ``totality_rejected_chunks``, depends on to be observable at all: an
    empty-``anon_facts`` chunk whose SOTA response still collided would
    otherwise be silently under-counted as "no rejection").  Every other
    exit (no orphans at all) also ``return []``, so callers can do a
    plain truthiness test on the result without a falsy-``None``-vs-falsy-
    ``[]`` trap.

    Only ONE production caller remains:
    :func:`~paramem.graph.cloud_egress.deanonymize_facts`, which runs
    this check UNCONDITIONALLY as step 1 (the primitive cannot be skipped
    from any of the three fact-deanonymizing paths — session-tier
    extraction, graph-tier enrichment, or their calibration harnesses —
    since none of them can reach ``_apply_bindings`` any other way) and
    REJECTS THE WHOLE DELTA on a non-empty verdict, returning
    ``DeanonResult(facts=[], verdict=verdict, ...)`` without substituting
    anything — the caller decides the fallback (e.g. the pre-enrichment
    local-extract facts).  The ``sota_bindings=None``, ``observed=None``,
    default ``diagnostic_key`` shape (kept as this function's default
    parameters for direct/unit-test use) has no production caller:
    anonymized facts are built by
    :func:`~paramem.graph.cloud_egress.anonymize_for_cloud` directly from
    ``graph.relations`` and the CORE map, so an orphan placeholder in a
    local fact is structurally impossible, not merely diagnosed and
    logged.  Matches both braced (``{Event_1}``) and bare (``Event_1``)
    placeholder forms via :data:`PLACEHOLDER_TOKEN_RE`.

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
    # Per-fact placeholder scan — skipped (not an early RETURN) when
    # there is nothing to scan, so a non-empty ``orphans`` set from the
    # collision branch above still reaches the single diagnostic-writing
    # exit point below rather than escaping unwritten.
    if anon_facts:
        resolvable = set(_resolution_map(reverse_mapping, sota_bindings or {}, observed))
        for f in anon_facts:
            if not isinstance(f, dict):
                continue
            for field in ("subject", "object"):
                for t in PLACEHOLDER_TOKEN_RE.findall(str(f.get(field, ""))):
                    name = t[0] or t[1]
                    if name not in resolvable:
                        orphans.add(name)
        # Q3 completeness: a binding value may itself contain a bare
        # placeholder (e.g. "Senior Engineer at Org_9") that never
        # resolves — the braced pass exposes it but nothing substitutes
        # it, so it drops silently.
        for value in (sota_bindings or {}).values():
            if not isinstance(value, str):
                continue
            for t in PLACEHOLDER_TOKEN_RE.findall(value):
                name = t[0] or t[1]
                if name not in resolvable:
                    orphans.add(name)
    if orphans:
        ordered = sorted(orphans)
        # A single message: the only production caller
        # (``deanonymize_facts``) always rejects the WHOLE delta on a
        # non-empty verdict, so "the enrichment delta will be rejected"
        # is accurate for every caller that reaches this branch in
        # practice; a direct/unit-test caller exercising the
        # ``sota_bindings=None`` default shape gets the same message.
        logger.warning(
            "Binding-totality violation: %d orphan/conflict placeholder(s) in "
            "anon_facts not resolvable: %s. The delta will be rejected.",
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


def invert_forward_mapping(mapping: dict) -> dict[str, str]:
    """Invert a forward ``{key: value}`` table to ``{value: key}``.

    Skips any entry whose key or value is not a ``str``. When multiple
    forward keys share the same value (a many-to-one forward map — e.g.
    two real names the model scrubbed onto the same placeholder), the
    FIRST key encountered in ``mapping``'s iteration order wins; later
    duplicates are silently dropped via ``setdefault``.

    THE only forward -> reverse inversion of the CORE anonymization table
    in this module — scoped precisely: a caller elsewhere inverting a
    DIFFERENT map for a DIFFERENT purpose (e.g.
    :func:`~paramem.graph.extractor._sota_pipeline` inverting
    ``scope.resolution`` — a placeholder -> real map, already the
    OUTPUT of :func:`~paramem.graph.cloud_egress.CloudScope.for_response`,
    not a raw forward table — for its own entity-type lookup) is not a
    second forward -> reverse inversion of THIS table and does not
    conflict with this claim.

    THE single caller for the CORE table is
    :func:`_build_anonymization_mapping`. Previously two call sites each
    inverted the CORE table independently with OPPOSITE tie-breaks: the
    session tier used first-wins (``reverse.setdefault(v, k)``) while the
    graph tier (the pre-unification ``_graph_enrich_with_sota``) used
    LAST-wins (a plain dict comprehension, where a later ``k`` for the
    same ``v`` silently overwrote an earlier one) — so a many-to-one
    forward map restored a DIFFERENT real name at each tier for the
    identical placeholder. Post-unification, the graph tier no longer
    inverts anything itself — it receives an already-built
    ``AnonymizedPayload.reverse`` from
    :func:`~paramem.graph.cloud_egress.anonymize_for_cloud`, which reaches
    this function through :func:`_build_anonymization_mapping` exactly
    like every other migrated path.
    """
    out: dict[str, str] = {}
    for k, v in mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        out.setdefault(v, k)
    return out


def _build_anonymization_mapping(
    llm_mapping: dict,
    *,
    speaker_name: str | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assemble the real → placeholder mapping from the model's mapping.

    The anonymizer LLM is the SOLE scope authority (it decides, against
    the operator's ``scrub`` categories, which real values need a
    placeholder — see :func:`~paramem.graph.extractor.
    anonymize_with_local_model` and the module the prompt lives in,
    ``configs/prompts/anonymization.txt``).  This builder does not
    re-judge that decision, and does not re-derive it from any other
    source (graph entities, entity types, a scope allowlist) — there is
    no second detector on this path.  Its
    job is exactly two things, both invariant maintenance rather than
    classification:

    1. **Speaker-anchor invariants.**  ``speaker{N}`` is an anonymized
       handle by construction (CLAUDE.md's "ONE lowercase ``speaker{N}``
       everywhere"), never a real name to be re-mapped.  A ``llm_mapping``
       KEY that is speaker-id-shaped
       (:func:`~paramem.graph.name_match.is_speaker_id`) — e.g. a
       hallucinated ``{"speaker0": "Person_1"}`` — is dropped outright,
       forward AND reverse.  A ``llm_mapping`` VALUE that is
       speaker-id-shaped — e.g. ``{"RealName": "speaker0"}``, the model
       scrubbing a real name onto the anchor — keeps the forward scrub
       (harmless, and the only thing standing between that real name and
       the cloud) but drops the reverse write: a reverse entry keyed on
       ``speaker0`` would restore that real name onto every
       speaker-subject fact.
    2. **Speaker-name seeding.**  When the runtime knows the speaker's
       display name and the model's mapping doesn't already cover it
       (the model never sees ``speaker_name`` as an explicit prompt
       field — it can only have named it if the name happened to occur
       in the transcript text it saw), reuse the model's own hint if it
       named the speaker (exact or full-name match, e.g. ``"Alex"`` or
       ``"Alex Rivera"`` → reuse ``Person_1``) or mint a fresh
       ``Person_N`` via :func:`mint_placeholder`.

    Every other entry in ``llm_mapping`` is trusted and merged in
    unconditionally — the model already decided it is in scope against
    ``scrub``; this builder does not re-gate, walk, or float a
    completeness floor under that decision.

    Callers build the anonymized fact array directly from
    ``graph.relations`` and this builder's forward map (subject/object
    substituted, predicate untouched) — never from the LLM's response,
    which carries no facts.

    Args:
        llm_mapping: Canonicalised ``{real_name: placeholder}`` mapping
            from :func:`_normalize_anonymization_mapping` — the model's
            own ``scrub``-scoped decision, trusted as-is.
        speaker_name: Runtime-known display name of the session's
            speaker.  When set, this name is guaranteed to be covered.

    Returns:
        ``(forward, reverse)`` — ``forward`` is the ``{real_name:
        placeholder}`` mapping that feeds :func:`_build_anon_facts` (in
        :mod:`paramem.graph.extractor`).  ``reverse`` is the one-to-one
        ``{placeholder: real_name}`` map consumed by the deanon path
        (:func:`deanonymize_text`, :func:`_apply_bindings`) — built by
        inverting ``forward`` via :func:`invert_forward_mapping`
        (first-wins tie-break on a many-to-one forward map), after
        dropping any entry whose VALUE is speaker-id-shaped (invariant 1
        above).
    """
    mapping: dict[str, str] = {}

    for k, v in llm_mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if is_speaker_id(k):
            continue
        mapping[k] = v

    # Speaker-name seeding: ensure the runtime-known speaker name is
    # covered, reusing the model's own hint when it named the speaker
    # (exact or full-name match) and minting a fresh Person_N only when
    # it did not.
    if speaker_name and speaker_name not in mapping:
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
        else:
            person_prefix = entity_type_to_prefix("person")
            fresh = mint_placeholder(mapping.values(), person_prefix)
            mapping[speaker_name] = fresh

    reverse = invert_forward_mapping({k: v for k, v in mapping.items() if not is_speaker_id(v)})

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
         means CORE UNSCOPED (every ``reverse`` entry is legal).  This
         default is a UNIT-TEST-ONLY sentinel: every production caller
         reaches this function exclusively through
         :func:`~paramem.graph.cloud_egress.deanonymize_facts`, which
         always passes ``scope.observed`` — a ``frozenset``, never
         ``None`` — on every cloud path.  The ``None`` default exists so
         the five pre-existing positional-only call sites in
         ``test_extraction_pipeline.py`` (and any other direct unit test
         of this primitive) keep passing unchanged as the CORE-unscoped
         regression net for the primitive itself.  A ``set``/``frozenset``
         means CORE SCOPED to it — see :func:`_resolution_map`.
         ``reverse`` wins on any key collision in EITHER mode (CORE
         PRECEDENCE) — deterministic entity names over SOTA-sourced
         values, never the reverse.

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
       remains the fail-closed backstop for cause (c). An anonymizer-stage
       leak is not among the live causes any more: :func:`_build_anon_facts`
       constructs the anon-stage fact array directly from ``graph.relations``
       and the mapping, so an orphan placeholder in a LOCAL fact is
       structurally impossible — the only source reaching this sweep is
       SOTA's *returned* facts.

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


def deanonymize_text(text: str, resolution: dict[str, str]) -> str:
    """Restore real names in cloud-returned text via a resolution map.

    ``resolution`` is a ``{placeholder: real_name}`` map — in production,
    always :meth:`~paramem.graph.cloud_egress.CloudScope.resolution` (the
    ``observed``-scoped output of :func:`_resolution_map`), never a raw
    ``reverse`` table.  The parameter is named ``resolution``, not
    ``reverse``, because that is genuinely what every production caller
    feeds it — the ONE caller is
    :func:`~paramem.graph.cloud_egress.deanonymize_response_text`. This
    function never inverts a forward map itself — that inversion is the
    producer's job (:func:`invert_forward_mapping`, inside
    :func:`_build_anonymization_mapping`).

    Word-boundary anchored, so a placeholder embedded in unrelated
    text doesn't match.  Idempotent on text without placeholders or
    with an empty resolution map.
    """
    if not text or not resolution:
        return text
    return _substitute_whole_words(text, resolution)
