"""The anonymize <-> deanonymize placeholder primitive kit.

Every cloud round trip that touches PII goes through this module's
primitives: mint a placeholder token, build the real-name <-> placeholder
table, detect declared/shaped tokens in text, resolve core + cloud-sourced
tables into one substitution map, substitute, and insert placeholders into
an already-serialized fact list. Nothing here has an opinion about WHERE
the table comes from (local anonymizer, cloud enrichment, cloud-egress
caller) or WHAT happens to the result (cloud prompt construction,
plausibility filtering, ``Relation`` construction) — those are
extraction-pipeline concerns that live in ``paramem.graph``.

Model-free and — by construction — free of any ``paramem.graph`` import:
every primitive here operates on plain ``dict``/``str`` fact artifacts,
never a ``Relation`` or ``SessionGraph``. Rendering a ``Relation`` into a
fact dict is the caller's job, in ``paramem/graph/``. One exception to
"IO-free": :func:`_build_anonymization_mapping` resolves the speaker
placeholder prefix via :func:`~paramem.config.taxonomy.entity_type_to_prefix`,
a cached (``lru_cache``) read of ``configs/schema.yaml`` — ``paramem.config``
is a leaf package with no graph/server dependency of its own, so this stays
within the "no ``paramem.graph`` import" boundary while resolving the
taxonomy directly rather than taking it as a caller-supplied parameter.

Token shape today is BARE (``Person_1``); a braced form (``{Person_1}``)
exists only for the in-text detection net and for the cloud's own
brace-binding mint protocol. Nothing here hardcodes the bare shape as
load-bearing — a future format flip only touches :func:`mint_placeholder`
and :data:`PLACEHOLDER_SHAPE_RE`.

Load-bearing invariants preserved from the pre-refactor implementation:

* ``observed is None`` in :func:`_resolution_map` means CORE UNSCOPED —
  never ``set()`` (an empty set would scope CORE to nothing and drop
  every fact on the paths that pass ``None``).
* CORE-LAST precedence: :func:`_resolution_map` merges ``cloud_bindings``
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

from paramem.config.taxonomy import entity_type_to_prefix
from paramem.utils.identity import is_speaker_id

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
PLACEHOLDER_SHAPE_RE = re.compile(rf"^{_BARE_PLACEHOLDER_SHAPE}$")

# Unanchored in-text detector — matches EITHER a braced mint
# (``{Event_1}``, prefix laxly ``\w+`` since the cloud's own mint protocol
# is the only producer of the braced form and doesn't always comply with
# PascalCase) OR a bare token (``Person_2``, strict PascalCase,
# word-boundary anchored). Consumed via :func:`_placeholder_tokens` (the
# ONE ``findall`` + name-extraction site) by :func:`_fact_tokens` and
# :meth:`~paramem.cloud.deanonymize.CloudScope.response`'s binding-value
# pruning; the residual sweep in :func:`_apply_bindings` uses the pattern
# directly for a boolean presence check (``.search``), not name
# extraction. (The ``observed`` legality domain is computed by
# :meth:`~paramem.cloud.deanonymize.CloudScope.response` from the
# DECLARED vocabulary by substring containment, never by this pattern.)
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


def placeholder_prefix(token: str) -> str | None:
    """Split a placeholder-shaped token into its prefix.

    ``"Home_Address_1"`` -> ``"Home_Address"``; ``"Person_2"`` ->
    ``"Person"``; ``None`` when *token* is not shaped per
    :data:`PLACEHOLDER_SHAPE_RE` (the anchored full-string validator —
    this function never partially matches an unshaped token).

    THE only place a shaped token is split into prefix + numeric index.
    Consumed by the cross-slice placeholder renumber in
    :func:`~paramem.cloud.anonymize.anonymize` (a collision between two
    slices' independently-minted placeholders is resolved by re-minting
    onto the same prefix via :func:`mint_placeholder`) — never
    re-implement the ``rsplit("_", 1)`` split at a second call site.
    """
    if not PLACEHOLDER_SHAPE_RE.match(token):
        return None
    return token.rsplit("_", 1)[0]


def braced(token: str) -> str:
    """Wrap a bare placeholder token in braces: ``"Person_1"`` -> ``"{Person_1}"``.

    THE only place a placeholder token is wrapped in braces. Used both
    to build the braced-literal substitution key in :func:`_apply_bindings`
    and to construct the cloud's braced-mint transcript notation in
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
    machine-minted placeholder tokens in the DEANONYMIZE direction (this
    same function is the whole of what a bare ``deanonymize_text(text,
    resolution)`` call would do — there is no separate wrapper). Folding
    would let a mapped person name silently consume its lowercase
    common-noun homograph (a person named "Bill" matching the electricity
    "bill"), and would let literal placeholder text resolve against a real
    name, defeating the fail-closed residual-token drop before it ever
    sees the token. Identity reconciliation — matching a mapping key
    to the fold graph's own canonical node-key text — is a separate step
    performed by the one caller that needs it, before this function ever
    sees the mapping — see
    :func:`~paramem.training.graph_enrich.enrich_graph`.
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


def insert_placeholders(facts: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Substitute ``subject``/``object`` through the forward ``mapping``,
    leaving every other fact-dict field verbatim.

    One dict per input fact: ``subject``/``object`` go through
    :func:`_substitute_whole_words`; every other key (``predicate``,
    ``relation_type``, ``confidence``, ``speaker_id``, or anything else a
    caller's fact dict happens to carry) is copied through unchanged via
    ``{**f, ...}`` — the predicate is NEVER a substitution target, so a
    placeholder cannot be glued into it at this stage (the motivating bug,
    ``language_proficiency_Language_3``, can still occur in the cloud's
    *returned* facts, which is why the deanon-stage predicate invariant in
    :func:`_apply_bindings` stays).

    Callers render their own artifact (a ``Relation`` list, a NetworkX
    subgraph's serialized triples, ...) into fact dicts BEFORE calling this
    — this primitive never reads a graph type, only ``dict``.  ``mapping``
    must already be the COMPLETE forward map (i.e.
    :func:`_build_anonymization_mapping`'s output) — a partial/HINT-only
    map here reproduces the exact leak this function exists to prevent.
    """
    return [
        {
            **f,
            "subject": _substitute_whole_words(str(f.get("subject", "")), mapping),
            "object": _substitute_whole_words(str(f.get("object", "")), mapping),
        }
        for f in facts
    ]


# ---------------------------------------------------------------------------
# Table normalize / validate — ONE normalizer, ONE validator, shared by the
# CORE anonymizer table and the cloud `bindings` table.
# ---------------------------------------------------------------------------


def _normalize_anonymization_mapping(
    mapping: dict, *, placeholder_side: str = "value"
) -> tuple[dict, dict]:
    """Normalize a table to canonical direction — placeholder shape on
    ``placeholder_side``.

    ``placeholder_side="value"`` (default) — the CORE anonymizer table,
    canonical direction ``{real_name: placeholder}``.
    ``placeholder_side="key"`` — the cloud ``bindings`` table, canonical
    direction ``{placeholder: real_text}`` — the OPPOSITE direction,
    since a binding maps a placeholder cloud minted to the real span it
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
    otherwise silently void real entities or cloud-minted entries).

    THE only normalizer for either table.

    THE CORE table (``placeholder_side="value"``, the default) has
    exactly ONE caller — :func:`~paramem.cloud.anonymize.anonymize`'s
    step 4 — and that caller's ``stats`` is a LIVE signal: it flows into
    ``AnonymizedContract.norm_stats`` and, for the session tier, into
    ``graph.diagnostics["mapping_ambiguous_dropped"]``.  The ``bindings``
    table variant (``placeholder_side="key"``) still has more than one
    caller (:meth:`~paramem.cloud.deanonymize.CloudScope.response`, and
    the unrelated legacy per-session delta parser
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
# Resolution map — core + secondary merge. CORE-LAST precedence and
# the `observed is None` CORE-UNSCOPED sentinel are load-bearing invariants.
# ---------------------------------------------------------------------------


def _resolution_map(
    reverse: dict[str, str],
    cloud_bindings: dict[str, str],
    observed: set[str] | None,
) -> dict[str, str]:
    """The ONE legality/resolution map — built once, consumed by
    :meth:`~paramem.cloud.deanonymize.CloudScope.response` (to compute
    ``resolution``/``core_resolution`` and to scope the binding-value
    pruning check) and by :func:`_apply_bindings` (as its substitution
    map).

    ``observed`` is ``None`` -> **CORE UNSCOPED** (today's behaviour;
    every deanon call that has no cloud-observed scope to pass — e.g. a
    call outside the cloud-enrichment cycle, or a unit test exercising
    :func:`_apply_bindings`/:func:`_resolution_map` directly): every
    ``reverse`` entry is legal, ``cloud_bindings`` is merged in
    underneath it. An empty-set default here instead of ``None``
    would scope CORE to nothing and drop every fact on those paths — see
    the callers' ``observed: set[str] | None = None`` declarations.

    ``observed`` is a ``set`` -> **CORE SCOPED** to it: a ``reverse``
    entry resolves only when cloud was actually shown its placeholder
    (``key in observed`` — a token in the rendered facts cloud saw, or in
    the anonymized transcript). Every ``cloud_bindings`` entry whose key
    is NOT in ``observed`` is cloud's own mint and resolves too. A key
    present in both domains is a CONFLICT — surfaced separately, purely
    informationally, by :func:`_binding_collisions` — but never gated on
    here: this map resolves it via CORE-LAST precedence regardless (the
    tie-break below), so a conflict is harmless whether or not the caller
    even runs the collision scan.

    **CORE PRECEDENCE (named invariant) — CORE-LAST BY CONSTRUCTION.**
    In both branches ``reverse`` entries are applied AFTER
    ``cloud_bindings``, so any key present in both resolves to CORE's
    value. Cloud can never override or misresolve against the core map.
    This is deliberate construction order, not an accident of dict-spread
    — do not reorder the two ``.update()`` calls below.
    """
    resolved: dict[str, str] = {}
    if observed is None:
        resolved.update(
            (k, v)
            for k, v in cloud_bindings.items()
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
            for k, v in cloud_bindings.items()
            if isinstance(k, str) and isinstance(v, str) and k and v and k not in observed
        )
        resolved.update(
            (k, v)
            for k, v in reverse.items()
            if isinstance(k, str) and isinstance(v, str) and k and v and k in observed
        )
    return resolved


def _placeholder_tokens(text: str) -> set[str]:
    """THE placeholder-token scan: every token name found in ``text``,
    braced (``{Event_1}``) or bare (``Event_1``).

    THE ONLY place :data:`PLACEHOLDER_TOKEN_RE.findall` plus the ``t[0] or
    t[1]`` braced/bare name-extraction pattern appears in the codebase.
    Every "which placeholder tokens appear in this string" question —
    whether over a fact field (:func:`_fact_tokens`), a cloud binding's
    own value (:meth:`~paramem.cloud.deanonymize.CloudScope.response`), or
    anything else — routes through this function. Never re-implement the
    ``findall`` + name-extraction loop at a second call site.
    """
    return {t[0] or t[1] for t in PLACEHOLDER_TOKEN_RE.findall(str(text))}


def _fact_tokens(fact: dict) -> set[str]:
    """Union of :func:`_placeholder_tokens` over ``fact``'s ``subject``/
    ``object`` fields.

    Only ``subject``/``object`` are scanned — a placeholder glued into
    ``predicate`` is a SEPARATE invariant, enforced by
    :func:`_apply_bindings` step 1, not this scan.
    """
    tokens: set[str] = set()
    for field in ("subject", "object"):
        tokens |= _placeholder_tokens(fact.get(field, ""))
    return tokens


def _fact_orphans(fact: dict, resolvable: set[str]) -> set[str]:
    """The set of unresolvable placeholder token names found in ``fact``'s
    ``subject``/``object`` fields — :func:`_fact_tokens` minus
    ``resolvable``.

    THE per-fact orphan predicate — used by
    :func:`~paramem.graph.extractor._apply_enrichment_delta` as the
    per-``add``/``modify`` accept/reject test, one fact at a time. Never
    duplicate this at a second call site.
    """
    return _fact_tokens(fact) - resolvable


def _binding_collisions(
    reverse_mapping: dict,
    *,
    cloud_bindings: dict | None = None,
    observed: set[str] | None = None,
) -> list[str]:
    """Collision scan: a ``cloud_bindings`` key that clashes with the CORE
    reverse map, or with the ``observed`` scope — ALWAYS informational,
    never a rejection signal.  A binding for a token cloud was SHOWN is
    inert under CORE-LAST precedence (:func:`_resolution_map` always
    resolves such a key to the CORE value); see that function's
    docstring for why a collision cannot corrupt resolution.

    When ``cloud_bindings`` is given and ``observed`` is a set, any
    ``cloud_bindings`` KEY that is also in ``observed`` is a CONFLICT
    (cloud referencing/rebinding something it was already shown as a core
    reference).  When ``observed`` is ``None`` (CORE unscoped), the scan
    instead flags any KEY present in both ``cloud_bindings`` and
    ``reverse_mapping`` with a DIFFERING value.

    Returns the sorted list of colliding keys, ``[]`` when the scan found
    nothing or ``cloud_bindings`` is empty/``None``.  Writes NOTHING to
    any caller-owned object: the ONE production caller,
    :func:`~paramem.cloud.deanonymize.deanonymize_facts`, writes the
    result onto its own ``DeanonResult.collisions``.
    """
    if not cloud_bindings:
        return []
    if observed is not None:
        collisions = sorted(k for k in cloud_bindings if k in observed)
    else:
        collisions = sorted(
            k for k, v in cloud_bindings.items() if k in reverse_mapping and reverse_mapping[k] != v
        )
    if collisions:
        logger.warning(
            "cloud binding collision: %d placeholder(s) present in both "
            "cloud_bindings and reverse_mapping with differing values "
            "(reverse_mapping wins): %s.",
            len(collisions),
            collisions[:5],
        )
    return collisions


# ---------------------------------------------------------------------------
# Detection — declared-vocabulary scan (substring, no \b, no regex) plus the
# shape regex above as the fail-closed net.
# ---------------------------------------------------------------------------


def _declared_placeholder_tokens(
    reverse: dict[str, str], cloud_bindings: dict[str, str] | None = None
) -> set[str]:
    """The declared placeholder-token vocabulary for a deanon call.

    Every key in ``reverse`` (the anonymizer's CORE map, ``placeholder ->
    entity_name``) plus every key in ``cloud_bindings`` (cloud's own minted
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
    if cloud_bindings:
        tokens.update(k for k in cloud_bindings if isinstance(k, str) and k)
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
    :func:`~paramem.graph.relation_build.apply_rebuild` inverting
    ``scope.resolution`` — a placeholder -> real map, already the
    OUTPUT of :func:`~paramem.cloud.deanonymize.CloudScope.response`,
    not a raw forward table — for its own entity-type lookup) is not a
    second forward -> reverse inversion of THIS table and does not
    conflict with this claim.

    THE single caller for the CORE table is
    :func:`_build_anonymization_mapping`.
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
    placeholder — see :func:`~paramem.cloud.anonymize.anonymize_transcript`
    and the module the prompt lives in,
    ``configs/prompts/anonymization.txt``).  This builder does not
    re-judge that decision, and does not re-derive it from any other
    source (graph entities, entity types, a scope allowlist) — there is
    no second detector on this path.  Its job is exactly two things, both
    invariant maintenance rather than classification:

    1. **Speaker-anchor invariants.**  ``speaker{N}`` is an anonymized
       handle by construction (CLAUDE.md's "ONE lowercase ``speaker{N}``
       everywhere"), never a real name to be re-mapped.  A ``llm_mapping``
       KEY that is speaker-id-shaped
       (:func:`~paramem.utils.identity.is_speaker_id`) — e.g. a
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
       placeholder via :func:`mint_placeholder`, prefixed via
       :func:`~paramem.config.taxonomy.entity_type_to_prefix` ("person")
       — resolved here directly rather than threaded in as a parameter:
       ``paramem.config`` is a leaf package (no ``paramem.graph``/
       ``paramem.server`` dependency), so this module can call it without
       crossing the ``paramem.cloud`` -> ``paramem.graph`` boundary the
       package docstring forbids.

    Every other entry in ``llm_mapping`` is trusted and merged in
    unconditionally — the model already decided it is in scope against
    ``scrub``; this builder does not re-gate, walk, or float a
    completeness floor under that decision.

    Callers build the anonymized fact array via
    :func:`insert_placeholders` from their own fact dicts and this
    builder's forward map (subject/object substituted, predicate
    untouched) — never from the LLM's response, which carries no facts.

    Args:
        llm_mapping: Canonicalised ``{real_name: placeholder}`` mapping
            from :func:`_normalize_anonymization_mapping` — the model's
            own ``scrub``-scoped decision, trusted as-is.
        speaker_name: Runtime-known display name of the session's
            speaker.  When set, this name is guaranteed to be covered.

    Returns:
        ``(forward, reverse)`` — ``forward`` is the ``{real_name:
        placeholder}`` mapping that feeds :func:`insert_placeholders`.
        ``reverse`` is the one-to-one ``{placeholder: real_name}`` map
        consumed by the deanon path (:func:`_apply_bindings`) — built by
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
            fresh = mint_placeholder(mapping.values(), entity_type_to_prefix("person"))
            mapping[speaker_name] = fresh

    reverse = invert_forward_mapping({k: v for k, v in mapping.items() if not is_speaker_id(v)})

    return mapping, reverse


# ---------------------------------------------------------------------------
# Deanonymization — the exit gate for facts, and the free-text deanon.
# ---------------------------------------------------------------------------

# The fields of a fact dict that constitute a `Relation` — exactly the
# keys read at the `Relation(**fact)`-equivalent construction site in
# `relation_build.build_relations` (subject/predicate/object/relation_type/confidence/
# symmetric; `speaker_id` is stamped separately from the session, never
# read off the fact). Any OTHER key on a fact dict (e.g. an `evidence`
# field an LLM invents) never reaches `Relation` and therefore cannot
# leak a placeholder anywhere observable — the residual sweep in
# `_apply_bindings` only tests these fields, and the cloud enrichment
# delta boundary (`_parse_enrichment_delta`) strips any other key from
# `add`/`modify` entries before they ever enter `enriched_anon`.
_FACT_FIELDS: frozenset[str] = frozenset(
    {"subject", "predicate", "object", "relation_type", "confidence", "symmetric"}
)


def _apply_bindings(
    facts: list[dict],
    reverse: dict[str, str],
    cloud_bindings: dict[str, str],
    observed: set[str] | None = None,
    *,
    resolution: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """De-anonymize facts via state-machine substitution — the SINGLE
    deanon exit gate, in three ordered steps:

    1. **Predicate invariant (BEFORE substitution).** A fact whose
       ``predicate`` field contains ANY token from
       :func:`_declared_placeholder_tokens` (``reverse`` keys union
       ``cloud_bindings`` keys), as a literal substring, is dropped
       outright — no splitting, no repair. This runs first so a
       poisoned predicate (``at_Org_1``) is never "resolved" into a
       garbage predicate (``at_Acme``): the predicate field is never a
       substitution target below, so checking it after substitution
       would find nothing wrong with an already-corrupted predicate.
    2. **Substitute** subject/object with :func:`_resolution_map`
       (``reverse``, ``cloud_bindings``, ``observed``), rendered in both
       braced and bare form:

       * **Anonymizer reverse map** (``reverse`` arg) —
         ``placeholder -> entity_name`` produced by
         :func:`_build_anonymization_mapping`.
       * **cloud bindings** (``cloud_bindings`` arg) —
         ``placeholder_name -> real_text`` that cloud emitted alongside
         its enriched facts (new entities cloud minted, e.g.
         ``Event_1``).
       * **``observed``** (trailing, defaulted ``None``) — ``None``
         means CORE UNSCOPED (every ``reverse`` entry is legal).  This
         default is a UNIT-TEST-ONLY sentinel: every production caller
         reaches this function exclusively through
         :func:`~paramem.cloud.deanonymize.deanonymize_facts`, which
         always passes ``scope.observed`` — a ``frozenset``, never
         ``None`` — on every cloud path.  A ``set``/``frozenset`` means
         CORE SCOPED to it — see :func:`_resolution_map`.
         ``reverse`` wins on any key collision in EITHER mode (CORE
         PRECEDENCE) — deterministic entity names over cloud-sourced
         values, never the reverse.

       The union round-trips a placeholder regardless of which form
       (braced or bare) it was actually emitted in: Cloud's contract asks
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
         a. Cloud introduced a braced placeholder but omitted its binding.
         b. Cloud emitted a bare placeholder that was never in the
            anonymizer mapping (anonymizer leak).
         c. Composite strings where one of multiple placeholders
            couldn't be resolved.
       Inside the full session flow
       (``paramem.graph.flows.SESSION_EXTRACT``, whose
       ``deanonymize`` stage is where this sweep actually runs), causes
       (a) and (b) are now mostly pre-empted upstream by
       :func:`~paramem.graph.extractor._apply_enrichment_delta`, which
       drops an unresolvable ``add`` and reverts an unresolvable
       ``modify`` individually (per-triple, not a whole-delta rejection)
       BEFORE the fact ever reaches this function — so a bad mint sheds
       only the one action that carried it.  This sweep remains the
       fail-closed backstop for whatever slips past that per-triple gate
       (in particular cause (c), and any fact this function is invoked on
       directly, outside the full session flow — e.g. its own unit
       tests). An anonymizer-stage leak is not among the live causes any
       more: :func:`insert_placeholders` constructs the anon-stage fact
       array directly from the caller's relations and the mapping, so an
       orphan placeholder in a LOCAL fact is structurally impossible — the
       only source reaching this sweep is cloud's *returned* facts.

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
    (``_extract_cloud_bindings``) which produced bogus mappings under
    multi-token replace blocks (bug 5).

    ``resolution`` — when the caller already holds
    :attr:`~paramem.cloud.deanonymize.CloudScope.resolution` (the ONE
    production shape, via :func:`~paramem.cloud.deanonymize.
    deanonymize_facts`), pass it here instead of letting step 2 recompute
    ``_resolution_map(reverse, cloud_bindings, observed)`` a second time
    from the same three inputs. ``None`` (default) preserves the original
    behaviour for direct/unit-test callers with no ``CloudScope`` to hand.
    """
    declared = _declared_placeholder_tokens(reverse, cloud_bindings)

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
    resolve = (
        resolution if resolution is not None else _resolution_map(reverse, cloud_bindings, observed)
    )
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
