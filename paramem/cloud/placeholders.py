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
"IO-free": :func:`build_forward_table` resolves the speaker
placeholder prefix via :func:`~paramem.config.taxonomy.entity_type_to_prefix`,
a cached (``lru_cache``) read of ``configs/schema.yaml`` — ``paramem.config``
is a leaf package with no graph/server dependency of its own, so this stays
within the "no ``paramem.graph`` import" boundary while resolving the
taxonomy directly rather than taking it as a caller-supplied parameter.

:class:`~paramem.cloud.anonymize_steps.ScanResult` is imported only under
``TYPE_CHECKING`` below — :func:`build_forward_table`'s type hint names it,
but ``paramem.cloud.anonymize_steps`` imports FROM this module
(:data:`_MAX_MAPPING_TEXT_CHARS`, :func:`_word_boundary_ok`),
so a runtime import here would cycle.

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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from paramem.config.taxonomy import entity_type_to_prefix
from paramem.utils.identity import canonical, is_speaker_id

if TYPE_CHECKING:
    from paramem.cloud.anonymize_steps import ScanResult

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
    this module — including :func:`build_forward_table`'s
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


def unbraced(token: str) -> str:
    """Inverse of :func:`braced`: strip one surrounding ``{...}`` pair.

    ``"{Person_1}"`` -> ``"Person_1"``; a string that does not both start
    with ``{`` and end with ``}`` is returned unchanged. No whitespace
    trimming — a real-name side is a substitution key and must survive
    byte-identical. Idempotent for the shapes this module produces: a
    bare token passes through unchanged, and a doubly-braced token
    (``"{{Person_1}}"``) has only its outer pair removed per call, so a
    second call on the already-stripped result is a no-op once no
    surrounding pair remains.

    THE only place a candidate table-entry string is stripped of its
    placeholder braces before shape validation — see
    :func:`_normalize_anonymization_mapping`.
    """
    if len(token) >= 2 and token.startswith("{") and token.endswith("}"):
        return token[1:-1]
    return token


# ---------------------------------------------------------------------------
# Substitution — word-boundary text substitution over a {key: value} table.
# ---------------------------------------------------------------------------


def _is_word_char(c: str) -> bool:
    """Match Python regex ``\\w`` semantics: alphanumeric (Unicode-aware) or underscore."""
    return c.isalnum() or c == "_"


def _word_boundary_ok(text: str, key: str, pos: int) -> bool:
    """Edge-aware whole-word boundary test for a candidate match of *key*
    at *pos* in *text*: a side needs a boundary check only when the KEY's
    edge char on that side is a word char (:func:`_is_word_char`) — see
    :func:`_substitute_whole_words`'s docstring for the full rule.

    THE one boundary predicate shared by :func:`_substitute_whole_words`
    (deciding whether a scanned candidate position is a real match, inside
    its replace-everywhere scan) and
    :func:`~paramem.cloud.anonymize_steps._scan_drop_reason` (the scan's
    own whole-word verification check) — never re-implement the edge-aware
    check at a second call site.
    """
    end = pos + len(key)
    if _is_word_char(key[0]) and pos > 0 and _is_word_char(text[pos - 1]):
        return False
    if _is_word_char(key[-1]) and end < len(text) and _is_word_char(text[end]):
        return False
    return True


def _substitute_whole_words_and_applied(
    text: str,
    mapping: dict[str, str],
) -> tuple[str, set[str]]:
    """The ONE longest-first, edge-aware-boundary replace-everywhere walk —
    shared by :func:`_substitute_whole_words` (its ``str``-only public
    form) and :func:`_applied_whole_word_keys` (its key-reporting form),
    each reading one half of this function's return rather than
    re-implementing the walk.

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

    Returns ``(substituted_text, applied_keys)`` — ``applied_keys`` is the
    subset of ``mapping``'s keys that matched at least one position; a key
    absent from it substituted nothing anywhere in ``text``.
    """
    if not mapping or not text:
        return text, set()
    normalized = {k: v for k, v in mapping.items() if isinstance(k, str) and k}
    if not normalized:
        return text, set()
    keys_sorted = sorted(normalized, key=len, reverse=True)

    parts: list[str] = []
    applied: set[str] = set()
    pos = 0
    n = len(text)
    while pos < n:
        matched = False
        for key in keys_sorted:
            klen = len(key)
            end = pos + klen
            if end > n:
                continue
            if text[pos:end] != key:
                continue
            if not _word_boundary_ok(text, key, pos):
                continue
            replacement = normalized[key]
            if not isinstance(replacement, str):
                continue
            parts.append(replacement)
            applied.add(key)
            pos = end
            matched = True
            break
        if not matched:
            parts.append(text[pos])
            pos += 1
    return "".join(parts), applied


def _substitute_whole_words(
    text: str,
    mapping: dict[str, str],
) -> str:
    """Replace whole-word occurrences of mapping keys with their values.

    Delegates to :func:`_substitute_whole_words_and_applied` (the shared
    longest-first, edge-aware-boundary walk) and returns only the
    substituted text — see that function's docstring for the full
    matching rule (edge-aware boundaries, longest-first, exact/
    case-sensitive matching).
    """
    substituted, _applied = _substitute_whole_words_and_applied(text, mapping)
    return substituted


def _applied_whole_word_keys(text: str, mapping: dict[str, str]) -> set[str]:
    """The reporting form of :func:`_substitute_whole_words`: which keys of
    *mapping* actually matched somewhere in *text*, without needing the
    substituted text itself.

    Runs the identical walk (:func:`_substitute_whole_words_and_applied`)
    over the same ``(text, mapping)`` pair and returns only the
    applied-keys half. A key this function does not return is INERT for
    that ``(text, mapping)`` pair: :func:`_substitute_whole_words` would
    substitute it nowhere. The one production caller
    (:func:`~paramem.cloud.anonymize.anonymize`) uses this to prune a
    forward table down to the keys that are live over one payload, before
    deriving ``reverse``/``declared``/``anon_transcript`` from it — see
    that function's docstring.
    """
    _substituted, applied = _substitute_whole_words_and_applied(text, mapping)
    return applied


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
    :func:`build_forward_table`'s output) — a partial/HINT-only
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

# Longest raw candidate string recorded in a dropped-entry payload — a
# diagnostic sample, not a redaction: the surviving substring is still
# whatever the caller's mapping entry actually was, just capped so one
# oversized entry cannot bloat a session snapshot.
_MAX_MAPPING_TEXT_CHARS: int = 64


def _dropped_mapping_entry(placeholder_side: str, key_str: str, value_str: str) -> dict:
    """Build one ``dropped_entries`` record for
    :func:`_normalize_anonymization_mapping`.

    ``side`` is the caller's declared ``placeholder_side`` — which half of
    the pair was supposed to carry the placeholder. ``text`` is that
    side's raw string (neither side matched the placeholder shape, so it
    is necessarily malformed on the declared side too), truncated to
    :data:`_MAX_MAPPING_TEXT_CHARS`. ``counterpart_len`` is a length only
    — the other (presumed real) side's content is never logged.
    """
    text, counterpart = (key_str, value_str) if placeholder_side == "key" else (value_str, key_str)
    return {
        "side": placeholder_side,
        "text": text[:_MAX_MAPPING_TEXT_CHARS],
        "counterpart_len": len(counterpart),
    }


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

    Each candidate side is passed through :func:`unbraced` before shape
    validation, so a braced candidate (``{Event_1}``, the cloud binding
    mint shape) validates exactly as its bare form would. Whichever side
    then matches :data:`PLACEHOLDER_SHAPE_RE` becomes ``placeholder_side``
    in the output, stored BARE (the unbraced form); the other (real) side
    is stored VERBATIM, braces and all — it is a substitution key and must
    match the transcript exactly. When BOTH sides match (a real-world name
    that happens to be PascalCase_N-shaped, e.g. ``GPT_4``, ``COVID_19``),
    the caller's declared ``placeholder_side`` breaks the tie — the entry
    is kept as-is, not dropped. Only NEITHER side matching (after
    unbracing) is genuinely ambiguous and dropped (logging, and recorded
    per-entry in ``dropped_entries`` — see the return contract below).

    ``placeholder_side="value"`` ALSO accepts a speaker-id-shaped VALUE
    (:func:`~paramem.utils.identity.is_speaker_id`) as placeholder-shaped,
    even though a speaker id never matches :data:`PLACEHOLDER_SHAPE_RE`
    (no PascalCase prefix, no underscore before the digit) — the
    fold-onto-token carve-in, historically what let a model-authored
    ``{"RealName": "speaker0"}`` entry survive normalization for the CORE
    anonymizer table. This carve-in is deliberately NOT extended to
    ``placeholder_side="key"`` — the cloud ``bindings`` table — where a
    speaker-id-shaped KEY stays genuinely ambiguous (a cloud model has no
    authority to bind new content onto the identity anchor).

    Current status: the local anonymizer no longer authors a
    ``{real: placeholder}`` mapping at all — SCAN only lists real values,
    and :func:`build_forward_table` mints every placeholder in code,
    including the fold-onto-``speaker{N}`` decision (from the ANCHOR call's
    ``anchor_names``). ``placeholder_side="value"`` (and this speaker-id
    carve-in) therefore has no production caller left; it is retained
    here, unnarrowed, since the ``bindings`` table (``placeholder_side="key"``)
    variant below still shares this same normalizer and its own callers
    and tests are unaffected.

    Returns ``(canonical_mapping, stats)`` where ``stats`` has
    ``{inverted, dropped, dropped_entries}`` — ``inverted``/``dropped`` are
    counts, ``dropped_entries`` is a ``list[dict]`` (see
    :func:`_dropped_mapping_entry`) with one record per dropped pair,
    ``len(dropped_entries) == dropped``. Surfaces the mapping-quality
    signal to callers so they can persist it in diagnostics
    (ambiguous-drop can otherwise silently void real entities or
    cloud-minted entries).

    THE only normalizer for either table.

    THE CORE table direction (``placeholder_side="value"``, the default)
    has NO production caller — see the "Current status" note above. The
    ``bindings`` table
    variant (``placeholder_side="key"``) still has more than one caller
    (:meth:`~paramem.cloud.deanonymize.CloudScope.response`, and the
    unrelated legacy per-session delta parser
    :func:`~paramem.graph.extractor._parse_enrichment_delta`) and its
    ``stats`` is not currently surfaced to a diagnostic by either.
    """
    if not mapping:
        return mapping, {"inverted": 0, "dropped": 0, "dropped_entries": []}
    out: dict = {}
    inverted = 0
    dropped_entries: list[dict] = []
    for k, v in mapping.items():
        k_str, v_str = str(k), str(v)
        k_bare, v_bare = unbraced(k_str), unbraced(v_str)
        k_match = bool(PLACEHOLDER_SHAPE_RE.match(k_bare))
        v_match = bool(PLACEHOLDER_SHAPE_RE.match(v_bare))
        if placeholder_side == "value" and not v_match and is_speaker_id(v_str):
            # A speaker id (e.g. "speaker0") never matches
            # PLACEHOLDER_SHAPE_RE (no PascalCase prefix, no underscore
            # before the digit) but IS a legitimate placeholder-shaped
            # VALUE under fold-onto-token anonymization: the
            # anonymizer LLM is authorized to emit {"RealName": "speaker0"}
            # when a coreferring real-name mention folds onto the user's
            # own token. Without this carve-in the entry has neither side
            # placeholder-shaped and is dropped as ambiguous below, so
            # historically the fix that let the CORE table builder actually
            # receive the entry — see the "Post-anonymizer-split status"
            # docstring note above (this direction has no production
            # caller left post-split).
            # Guarded to ``placeholder_side="value"`` only: the cloud
            # ``bindings`` table (``placeholder_side="key"``) must never
            # treat a speaker id as a valid KEY placeholder — a cloud model
            # is not authorized to bind new content onto the identity
            # anchor, so a binding shaped that way stays genuinely
            # ambiguous/rejected, not silently accepted.  (No `v_bare`
            # reassignment needed here: a speaker id is never braced, so
            # `unbraced(v_str)` above already left `v_bare == v_str`.)
            v_match = True

        # Which side the shape test says is the placeholder; a tie (both
        # match — e.g. a real-world name that also happens to be
        # PascalCase_N-shaped, like "GPT_4") is broken toward the
        # caller's declared side, not dropped.
        if k_match and v_match:
            placeholder_on_key = placeholder_side == "key"
        elif k_match:
            placeholder_on_key = True
        elif v_match:
            placeholder_on_key = False
        else:
            # Neither side matches the placeholder shape (even after
            # unbracing) — we cannot tell which side is the placeholder.
            # Dropping is safer than keeping: retaining would corrupt the
            # resolution map with a real-to-real entry.
            dropped_entries.append(_dropped_mapping_entry(placeholder_side, k_str, v_str))
            continue

        # Write in the canonical direction: the placeholder side stored
        # bare, the real side stored verbatim.
        if placeholder_on_key:
            placeholder_str, real_str = k_bare, v_str
        else:
            placeholder_str, real_str = v_bare, k_str
        if placeholder_on_key != (placeholder_side == "key"):
            inverted += 1
        if placeholder_side == "key":
            out[placeholder_str] = real_str
        else:
            out[real_str] = placeholder_str
    if inverted:
        logger.info(
            "Anonymization table: inverted %d/%d pair(s) to canonical "
            "direction (placeholder_side=%r)",
            inverted,
            len(mapping),
            placeholder_side,
        )
    if dropped_entries:
        logger.warning(
            "Anonymization table: dropped %d/%d ambiguous pair(s) (neither "
            "side matches the placeholder shape, even after unbracing); "
            "affected entries will not resolve.",
            len(dropped_entries),
            len(mapping),
        )
    return out, {
        "inverted": inverted,
        "dropped": len(dropped_entries),
        "dropped_entries": dropped_entries,
    }


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


def _whole_word_contains(haystack: str, needle: str) -> bool:
    """True when *needle* occurs as a whole-word substring of *haystack*,
    at any position — the same edge-aware boundary rule
    :func:`_word_boundary_ok` applies at substitution time, reused here so
    "does this surface occur as a real sub-phrase of that one" and "would
    substitution actually match it" are the same question ON THE SAME
    INPUT SHAPE :func:`_substitute_whole_words` sees. The ONE caller
    (:func:`build_forward_table`'s surface-containment pass) always passes
    *haystack*/*needle* already folded through
    :func:`~paramem.utils.identity.canonical` ``mode="spaces"``, where a
    blank/underscore run collapses to a single space — so this test is
    deliberately BROADER than a raw substitution match on the original
    surfaces: ``"Elena_Varga"`` contains ``"Elena"`` here (both fold to
    ``"elena varga"`` / ``"elena"``) even though the literal string
    ``"Elena_Varga"`` does not contain the literal substring ``"Elena"``
    at a substitution-time word boundary. Empty *needle* and
    *needle == haystack* never match — this is a strict SUB-string test,
    not equality.
    """
    if not needle or needle == haystack:
        return False
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos == -1:
            return False
        if _word_boundary_ok(haystack, needle, pos):
            return True
        start = pos + 1


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

    THE single production caller for the CORE table is
    :func:`~paramem.cloud.anonymize.anonymize`, which inverts its own
    RECONCILED ``forward`` (:func:`build_forward_table`'s output, after
    identity reconciliation) into ``reverse`` — :func:`build_forward_table`
    itself no longer calls this function; see its own docstring's Returns
    section.
    """
    out: dict[str, str] = {}
    for k, v in mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        out.setdefault(v, k)
    return out


def build_forward_table(
    scans: "Sequence[ScanResult]",
    *,
    anchor_names: frozenset[str],
    speaker_id: str | None,
    speaker_name: str | None,
) -> dict[str, str]:
    """Assemble the real -> placeholder forward table from verified SCAN
    results — code mints every placeholder now; there is no model-authored
    mapping to normalize or invert (see the module docstring's "Model-free"
    note and :mod:`paramem.cloud.anonymize_steps`).

    THE one table constructor for the local-anonymizer CORE map. This
    function does the minting itself (:func:`mint_placeholder`) since the
    tagger never proposes placeholder values at all — SCAN only lists real
    values.

    Processing order: *scans* in the caller's given order — the resolved
    :class:`~paramem.config.taxonomy.ScrubCategory` tuple's own order
    (schema row order), the same order
    :func:`~paramem.cloud.anonymize_steps.scan_values` returns one result
    per — each category's own
    :attr:`~paramem.cloud.anonymize_steps.ScanResult.values` in
    first-occurrence order (already established by that function's own
    verification). Cross-category ties (the identical or a canonically
    equal real value scanned in two categories) resolve to the EARLIER
    category's placeholder (first-category-wins). The tie is over which
    placeholder PREFIX a value gets, and category order is the operator's
    own schema-row order — a stable, config-owned precedence, not an
    accident of "no shared offset index": every span already carries a
    coordinate in one global payload offset space (the tagger's), so true
    first-occurrence order across categories IS reconstructible here, and
    is deliberately NOT what this function uses — deciding the prefix by
    incidental payload position would let the same value mint ``Person_1``
    in one session and ``Profile_1`` in the next, purely because a scan
    order or a paraphrase shuffled which category's span came first.

    Canonical equality (:func:`~paramem.utils.identity.canonical`) decides
    placeholder SHARING, never deletion: a second surface that is only
    canonically equal to an already-minted one (``"Lena"`` after
    ``"lena"``) still becomes its OWN forward-table key, mapped onto the
    SAME placeholder the first canonically-equal surface minted — never a
    dropped/discarded entry, which would egress verbatim once mint-side
    verification stopped folding surfaces together
    (:func:`~paramem.cloud.anonymize_steps.scan_values` keeps every
    distinct verbatim surface the tagger returns; see that module's
    docstring). :func:`_substitute_whole_words` stays exact and
    case-sensitive by design — folding there would let a person named
    ``"Bill"`` consume the common noun ``"bill"``.

    **Surface containment** (applied AFTER the loop above and the
    speaker-name seeding below, so an anchor fold onto *speaker_id* is
    never disturbed): within one category, canonically-equal surfaces
    (:func:`~paramem.utils.identity.canonical`, the same fold the mint
    loop above uses to decide sharing) are ONE surface for this decision
    too — a group minted together (``"lena"``/``"Lena"``) is judged and
    re-pointed as a whole, never split so that one case- or
    diacritic-differing member follows a container while its sibling is
    left behind on the group's old placeholder. A group is judged exactly
    once, the first time the longest-canonical-form-first sweep (below)
    reaches any of its members: every member's own case/diacritic-
    preserving canonical form (:func:`_whole_word_contains`, on
    :func:`~paramem.utils.identity.canonical` ``mode="spaces"``) is
    tested for whole-word containment against every other surface in the
    category, and the matches are pooled across the whole group — one
    member matching is enough for the group to find a container even
    when a sibling's own case-differing form would not on its own
    (``"lena"`` alone never whole-word-matches the capitalized ``"Lena
    Marie Fischer"``, but its group-mate ``"Lena"`` does, and the pooled
    result carries ``"lena"`` along with it). A group pooled onto
    EXACTLY ONE other surface's PLACEHOLDER (containers that are only
    case/diacritic variants of each other, e.g. ``"Elena Varga"`` and
    ``"Elena VARGA"``, already share one placeholder from the mint loop
    above and so count as one, not two) shares that container's
    placeholder — a name fragment and the value it is part of are one
    entity, not two (``"Varga"``/``"Elena"`` alongside ``"Elena
    Varga"``). Every member stays its own forward-table key, so
    substitution still replaces every one of them. A group pooled onto
    TWO OR MORE distinct placeholders is genuinely ambiguous
    (``"Elena"``/``"elena"`` inside both ``"Elena Varga"`` and ``"Elena
    Fischer"``) and keeps its own freshly-minted placeholder rather than
    guessing which one it belongs to. A value already folded onto
    *speaker_id* by the anchor invariant below is skipped outright — this
    rule only reassigns a group that was otherwise minted its own
    placeholder. The rule is applied to each group exactly once, in the
    one category that actually minted its placeholder (first-category-
    wins, same as the mint loop) — a value merely co-listed in a second
    category's scan (the identical surface tagged under two labels, e.g.
    an email address tagged both ``person`` and ``email``) is never
    re-pointed a second time onto that second category's own container.

    Within a category, groups are judged LONGEST canonical form first
    (by the length of whichever member is reached first in that sweep,
    stable on a length tie, so an unresolved tie keeps the scan's own
    order) rather than in scan/payload order: a group's containers must
    already be settled — carrying their OWN final placeholder, including
    any re-point a still-longer container gave them — before that group's
    own ambiguity is decided, or "distinct placeholders among the
    containers" would not actually mean "distinct entities". This makes a
    transitive chain (``"Ann"`` inside ``"Ann Marie"`` inside ``"Ann Marie
    Bell"``) collapse onto one placeholder deterministically: the longest
    surface is judged first (it has no longer container, so it is never
    itself re-pointed), then ``"Ann Marie"`` sees a single container and
    shares its placeholder, then ``"Ann"`` sees that same single
    (already-repointed) placeholder on both its containers and shares it
    too — the SAME result regardless of the order *scan.values* happened to
    list the three surfaces in, since judgment order is now a function of
    canonical length, not payload position. A re-pointed group's members
    are each DELETED then RE-INSERTED into ``forward`` rather than
    reassigned in place, so every one of them sits AFTER its container in
    insertion order — production recomputes ``reverse`` from ``forward``
    via :func:`invert_forward_mapping`'s first-wins-by-insertion-order
    rule, so without the reorder a short fragment mentioned earlier in the
    payload could win the reverse map's tie-break over the full-length
    surface it was folded onto. One residual: when EVERY span of a real
    surface the tagger found is contained in a longer one (e.g. an
    over-long address span that contains the true address), the true
    surface never becomes its own forward-table key, and any occurrence
    of it the tagger did not separately mark egresses verbatim — the
    over-long key still covers the region the tagger DID mark.

    Two invariants:

    1. **Speaker-anchor invariants.**  ``speaker{N}`` is an anonymized
       handle by construction (CLAUDE.md's "ONE lowercase ``speaker{N}``
       everywhere"), never a real name to be re-mapped. A scanned value
       that is itself speaker-id-shaped
       (:func:`~paramem.utils.identity.is_speaker_id`) is dropped outright
       (defensive: :func:`~paramem.cloud.anonymize_steps.scan_values`'s own
       verification already drops it before it ever reaches here). A value
       in *anchor_names* — the model's ANCHOR-call decision, already
       restricted to the scanned surfaces by
       :func:`~paramem.cloud.anonymize_steps.ask_speaker_anchor` — folds
       onto *speaker_id* directly (the forward scrub, harmless and the only
       thing standing between that real name and the cloud) rather than a
       minted placeholder; such an entry is EXCLUDED from ``reverse`` (a
       reverse entry keyed on ``speaker_id`` would restore that real name
       onto every speaker-subject fact).
    2. **Speaker-name seeding.**  When the runtime knows the speaker's
       display name and it isn't already covered by a scanned/anchored
       entry, reuse an already-built entry's placeholder when SOME scanned
       surface names the speaker (exact or full-name match, e.g. ``"Alex"``
       or ``"Alex Rivera"`` -> reuse its placeholder) or mint a fresh
       placeholder via :func:`mint_placeholder`, prefixed via
       :func:`~paramem.config.taxonomy.entity_type_to_prefix` ("person").

    Every other scanned value is trusted and minted unconditionally — the
    model already decided it is in scope against ``scrub``; this builder
    does not re-gate, walk, or float a completeness floor under that
    decision.

    Callers build the anonymized fact array via :func:`insert_placeholders`
    from their own fact dicts and this builder's forward map (subject/
    object substituted, predicate untouched) — never from the model's
    response, which carries no facts.

    Args:
        scans: One :class:`~paramem.cloud.anonymize_steps.ScanResult` per
            configured category, in category order.
        anchor_names: The self-introduced subset of the scan union, from
            :func:`~paramem.cloud.anonymize_steps.ask_speaker_anchor` —
            already restricted to values that were actually scanned.
        speaker_id: THIS session's own well-shaped ``speaker{N}`` token, or
            ``None`` (the graph tier, which has no single-session speaker).
            A non-empty *anchor_names* with ``speaker_id=None`` is
            defensive-only (the caller's own three-way precondition —
            ``paramem.cloud.anonymize.anonymize``'s docstring — means this
            never happens in production): such a value mints an ordinary
            placeholder instead of folding, rather than raising.
        speaker_name: Runtime-known display name of the session's speaker.
            When set AND not itself speaker-id-shaped, this name is
            guaranteed to be covered. A speaker-id-shaped value here
            (:func:`~paramem.utils.identity.is_speaker_id`) is never seeded
            — see the reinstatement-bug comment at the seeding site.

    Returns:
        ``forward`` — the ``{real_name: placeholder}`` mapping that feeds
        :func:`insert_placeholders`. This function does NOT also return a
        ``reverse`` map: the one production caller
        (:func:`~paramem.cloud.anonymize.anonymize`) needs ``reverse``
        only AFTER cross-slice identity reconciliation has finished
        mutating ``forward``, so it recomputes ``reverse`` itself, once,
        via :func:`invert_forward_mapping` over the RECONCILED ``forward``
        (first-wins tie-break on a many-to-one forward map, after dropping
        any entry whose VALUE is speaker-id-shaped — invariant 1 above). A
        caller that needs ``reverse`` from this function's own
        pre-reconciliation ``forward`` computes it the same way:
        ``invert_forward_mapping({k: v for k, v in forward.items() if not
        is_speaker_id(v)})``.
    """
    forward: dict[str, str] = {}
    canon_to_placeholder: dict[str, str] = {}

    for scan in scans:
        for value in scan.values:
            if not isinstance(value, str) or not value or is_speaker_id(value):
                continue
            if value in forward:
                # Already minted for this exact surface — a scan
                # deduplicates within its own category, but two
                # categories can still list the identical string.
                continue
            if speaker_id and value in anchor_names:
                forward[value] = speaker_id
                canon_to_placeholder.setdefault(canonical(value), speaker_id)
                continue
            canon = canonical(value)
            shared = canon_to_placeholder.get(canon)
            if shared is not None:
                # Canonical SHARING, never deletion — see the docstring.
                forward[value] = shared
                continue
            placeholder = mint_placeholder(forward.values(), scan.category.prefix)
            forward[value] = placeholder
            canon_to_placeholder[canon] = placeholder

    # Speaker-name seeding. ``not is_speaker_id(speaker_name)`` guards
    # against the reinstatement bug: a display name that is itself
    # literally speaker-id-shaped must never seed a forward-map key.
    if speaker_name and not is_speaker_id(speaker_name) and speaker_name not in forward:
        speaker_lower = speaker_name.lower()
        reused: str | None = None
        for key, placeholder in forward.items():
            key_lower = key.lower()
            if key_lower == speaker_lower or key_lower.startswith(speaker_lower + " "):
                reused = placeholder
                break
        if reused is not None:
            forward[speaker_name] = reused
        else:
            fresh = mint_placeholder(forward.values(), entity_type_to_prefix("person"))
            forward[speaker_name] = fresh

    # Surface containment — placeholder SHARING within one category, never
    # deletion, never a second minted key. Runs LAST, after every anchor
    # fold and the speaker-name seeding above, so a value already folded
    # onto ``speaker_id`` is left untouched (skipped below) rather than
    # reassigned onto some other surface's placeholder. See the docstring.
    #
    # ``value_mint_category`` records, for each scanned value, the index of
    # the category that actually minted its placeholder — first-category-
    # wins, mirroring the mint loop above (a value can appear in more than
    # one category's ``scan.values`` when the tagger returns the identical
    # surface under two labels, but its forward-table entry is owned by
    # exactly one category). The sharing rule below is applied to a value
    # ONLY in its owning category's pass — running it again in a category
    # the value merely co-occurs in would re-point an already-settled
    # placeholder onto that second category's own (unrelated) container.
    value_mint_category: dict[str, int] = {}
    for idx, scan in enumerate(scans):
        for v in scan.values:
            if isinstance(v, str) and v and not is_speaker_id(v):
                value_mint_category.setdefault(v, idx)

    for idx, scan in enumerate(scans):
        category_values = [
            v for v in scan.values if isinstance(v, str) and v and not is_speaker_id(v)
        ]
        canon_forms = {v: canonical(v, mode="spaces") for v in category_values}
        # Longest-canonical-form-first, stable on ties: every container a
        # shorter surface could fold onto has already been resolved (its
        # own re-point, if any, already applied) by the time that shorter
        # surface is judged — see the docstring's "Surface containment"
        # paragraph for why payload order must not decide the outcome.
        category_values = sorted(category_values, key=lambda v: len(canon_forms[v]), reverse=True)
        judged_placeholders: set[str] = set()
        for value in category_values:
            if value_mint_category.get(value) != idx:
                continue
            placeholder = forward.get(value)
            if placeholder == speaker_id:
                continue
            if placeholder in judged_placeholders:
                # This value's canonical-equality group (minted together,
                # sharing `placeholder`) was already judged via an earlier
                # member reached by this same longest-first sweep — a
                # group is judged exactly once so it can never be split
                # across two containment outcomes.
                continue
            judged_placeholders.add(placeholder)
            # The canonical-equality group `placeholder` already covers —
            # every member moves together, whatever this pass decides.
            group = [v for v in category_values if forward.get(v) == placeholder]
            # Pool containment matches across every member's own
            # case/diacritic-preserving form: a case-differing sibling
            # that would not itself whole-word-match a container (`"lena"`
            # against `"Lena Marie Fischer"`) still rides along when
            # ANOTHER member of the same group does match (`"Lena"`).
            containers: list[str] = []
            seen_containers: set[str] = set()
            for member in group:
                own_canon = canon_forms[member]
                for other in category_values:
                    if other in seen_containers or forward.get(other) == placeholder:
                        continue
                    if len(canon_forms[other]) > len(own_canon) and _whole_word_contains(
                        canon_forms[other], own_canon
                    ):
                        containers.append(other)
                        seen_containers.add(other)
            # Ambiguity is decided on DISTINCT PLACEHOLDERS, not raw
            # surface count: ``canon_to_placeholder`` already folds
            # case/diacritic variants of the same real value onto one
            # placeholder in the mint loop above, so two such variants
            # both containing the group (``"Elena Varga"`` /
            # ``"Elena VARGA"``) are not a genuine two-way ambiguity —
            # they name the same entity and share the same placeholder
            # already.
            container_placeholders = {forward[c] for c in containers}
            if len(container_placeholders) == 1:
                shared_placeholder = forward[containers[0]]
                # Re-point AND re-order EVERY member of the group: delete
                # then re-insert so each key sits AFTER its container in
                # ``forward``'s insertion order. Production recomputes
                # ``reverse`` from ``forward`` via
                # :func:`invert_forward_mapping`'s first-wins-by-
                # insertion-order rule (:func:`~paramem.cloud.anonymize.
                # anonymize`, after cross-slice reconciliation) — without
                # the delete+re-insert, re-assigning ``forward[member]`` in
                # place would keep a FRAGMENT's original (earlier)
                # insertion position, so it — not the full-length
                # container — would win the reverse map's first-wins
                # tie-break and become the surface that egresses on
                # de-anonymization.
                for member in group:
                    del forward[member]
                    forward[member] = shared_placeholder

    return forward


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
         :func:`build_forward_table`.
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
