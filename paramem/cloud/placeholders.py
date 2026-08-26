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
"IO-free": :func:`build_forward_table` resolves the person-category prefix
via :func:`~paramem.config.taxonomy.entity_type_to_prefix`, a cached
(``lru_cache``) read of ``configs/schema.yaml`` — ``paramem.config`` is a
leaf package with no graph/server dependency of its own, so this stays
within the "no ``paramem.graph`` import" boundary while resolving the
taxonomy directly rather than taking it as a caller-supplied parameter.

:class:`~paramem.cloud.anonymize_steps.ScanResult` is imported only under
``TYPE_CHECKING`` below — :func:`build_forward_table`'s type hint names it,
but ``paramem.cloud.anonymize_steps`` imports FROM this module
(:data:`_MAX_MAPPING_TEXT_CHARS`, :func:`_word_boundary_ok`),
so a runtime import here would cycle.

The minted token shape is BARE (``Person_1``); a braced form
(``{Person_1}``) exists only for the in-text detection net and for the
cloud's own brace-binding mint protocol. Nothing here hardcodes the bare
shape as load-bearing — the shape is declared by :func:`mint_placeholder`,
recognised by :data:`PLACEHOLDER_SHAPE_RE`, and read backwards by
:func:`_decompose_token` here and by
:func:`~paramem.config.taxonomy.placeholder_entity_type` — the sites a
format change touches.

Load-bearing invariants:

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
* MINT-LAST: :func:`build_forward_table` prunes before it mints, so every
  minted placeholder number is carried by at least one surviving key in
  ``forward`` — a prefix's numbers run ``1..N`` with no holes left by a
  containment merge or an inert-key drop.
* :func:`substitute_declared_renderings` builds its alternation from the
  handed vocabulary's keys only, so a token never minted for the contract
  is never matched; the declared vocabulary is distinct under the
  rendering equivalence :func:`_rendering_fold` defines.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from paramem.config.taxonomy import entity_type_to_prefix
from paramem.utils.identity import canonical, is_speaker_id

_V = TypeVar("_V")

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
# DECLARED vocabulary by whole-word containment, never by this pattern.)
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

    THE only place a placeholder token string is built. The one call
    site is :func:`build_forward_table`'s mint & emit pass, one call per
    surviving non-speaker group — never re-implement a local counter or
    scanning closure at a call site.
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

    THE one boundary predicate for every whole-word decision in the
    anonymize chain: :func:`_substitute_whole_words` (is a scanned candidate
    position a real match), :func:`~paramem.cloud.anonymize_steps._scan_drop_reason`
    (the scan's own whole-word verification), the HA leg's retained-surface
    filter in :mod:`paramem.server.egress` (does a key occur outside a
    retained span), :func:`substitute_declared_renderings` (does a
    rendering candidate, matched on the span as written, hold at that
    edge), and :func:`_whole_word_contains` (the containment pass in
    :func:`build_forward_table`) — never re-implement the edge-aware check
    at another call site.
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
    that side and can therefore match starting or ending mid-run — every
    character position is a candidate match start, not only word-char
    starts.  A key that IS word-char-bounded on a side (``"Bill"``)
    still requires a non-word (or string-edge) neighbour there, so
    ``"Bill"`` matches standalone but never inside ``"Billing"``.

    Longest keys are tried first at each position so multi-word keys
    preempt single-word prefixes (``"Person_2"`` before ``"Person"``).
    Empty / non-string keys are skipped defensively — local extractors
    occasionally emit ``null`` mapping entries.  Matching is
    case-sensitive — every call site's mapping keys are exact-case
    entity names or placeholder tokens.

    Matching is EXACT — never case-, separator-, or diacritic-folded — for
    both directions this walk serves. In the ANONYMIZE direction the keys
    are literal, verbatim human surfaces (real entity names), so
    exactness is what stops a mapped person name from silently consuming
    its lowercase common-noun homograph (a person named "Bill" matching
    the electricity "bill"). In the fact gate's DEANON-direction
    substitution (:func:`_apply_bindings`) the keys are machine-minted
    placeholder tokens, and exactness keeps a mangled or re-rendered token
    visible as a literal substring to that gate's fail-closed residual
    sweep — a tolerant substitution here would resolve such a token away
    before the sweep ever saw it. The prose gate
    (:func:`~paramem.cloud.deanonymize.deanonymize_text`) restores through
    :func:`substitute_declared_renderings` instead — the tolerant
    rendering walk, matched over shown tokens only, where a prose
    coincidence ("Person 1 of 3" with ``Person_1`` shown) restoring is
    accepted by design. Identity reconciliation — matching
    a mapping key to the fold graph's own canonical node-key text — is a
    separate step performed by the one caller that needs it, before this
    function ever sees the mapping — see
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


def _applied_whole_word_keys(text: str, keys: Iterable[str]) -> set[str]:
    """The reporting form of :func:`_substitute_whole_words`: which of
    *keys* actually matches somewhere in *text*, without needing the
    substituted text itself or a value to substitute in.

    Builds ``dict.fromkeys(keys, "")`` and runs the identical walk
    (:func:`_substitute_whole_words_and_applied`) over it, returning only
    the applied-keys half — the values are never read, so a caller with no
    mapping to hand (a set of candidate surfaces, not yet paired with a
    placeholder) can still ask "which of these substitutes somewhere in
    this text". A key this function does not return is INERT for that
    ``(text, keys)`` pair: :func:`_substitute_whole_words` would
    substitute it nowhere. Two production callers:
    :func:`build_forward_table`'s prune pass uses this to keep only the
    forward-table keys that are live over one payload before minting;
    :meth:`~paramem.cloud.deanonymize.CloudScope.response` uses it to
    scope ``observed`` to the declared tokens actually shown.
    """
    mapping = dict.fromkeys(keys, "")
    _substituted, applied = _substitute_whole_words_and_applied(text, mapping)
    return applied


def _decompose_token(token: str) -> tuple[str, str] | None:
    """Split *token* into ``(prefix, number)`` — the mint's own
    ``f"{prefix}_{n}"`` format (:func:`mint_placeholder`) read backwards,
    not an independent spelling of the shape.

    ``prefix, sep, number = token.rpartition("_")``, accepted only when
    *sep* is present, ``number.isdecimal()``, and re-minting
    ``f"{prefix}_{int(number)}"`` reproduces *token* exactly — so a
    zero-padded (``Person_01``), non-numeric (``Foo_Bar``), or
    separator-free (``speaker1``) tail does not decompose.
    ``isdecimal()`` (not ``isdigit()``) is what makes the following
    ``int()`` call total: ``isdigit()`` accepts non-decimal Unicode digits
    (e.g. a superscript ``"²"``) that ``int()`` itself rejects, which
    would raise instead of returning ``None``. Returns ``None`` for a
    token outside that domain.

    Composed by :func:`substitute_declared_renderings` (the rendering
    alternation) and by :func:`_rendering_fold` (the rendering
    equivalence) — the one place the mint's format is read backwards.
    """
    prefix, sep, number = token.rpartition("_")
    if not sep or not number.isdecimal():
        return None
    if f"{prefix}_{int(number)}" != token:
        return None
    return prefix, number


def substitute_declared_renderings(text: str, mapping: dict[str, str]) -> str:
    """Restore every rendering of a declared placeholder token in *text*
    to *mapping*'s real value — the reply-side counterpart to
    :func:`_substitute_whole_words`'s byte-exact outbound walk.

    A rendering of a token is the minted form, any casing of it, and the
    ``_`` between prefix and number written as one space — the domain the
    external service may re-case or re-space when it writes a token back
    in its reply; the domain stops there deliberately.

    Builds one case-insensitive alternation from *mapping*'s keys only:
    per key, :func:`_decompose_token` — a key that does not decompose
    contributes no alternative; a decomposing key contributes
    ``re.escape(prefix) + "[ _]" + number``, so only the final separator
    is flexible and a multi-segment prefix keeps its internal underscores
    literal. Alternatives are sorted longest-token-first (regex
    alternation is leftmost-alternative, so the sort is what makes a
    longer token like ``Person_10`` beat a shorter one like ``Person_1``
    at the same starting position). Candidates come from ``finditer``,
    left to right, non-overlapping; each is kept only when
    :func:`_word_boundary_ok` holds on the span as written. The result is
    assembled from the kept spans and the untouched slices between them,
    so a substituted real value is never rescanned.

    Returns *text* unchanged when *mapping* is empty or none of its keys
    decompose.
    """
    alternatives: list[tuple[str, str]] = []
    for token in mapping:
        decomposed = _decompose_token(token)
        if decomposed is None:
            continue
        prefix, number = decomposed
        alternatives.append((token, f"{re.escape(prefix)}[ _]{number}"))
    if not alternatives:
        return text
    alternatives.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    tokens_by_group = [token for token, _pattern in alternatives]
    combined = "|".join(f"({pattern})" for _token, pattern in alternatives)
    matcher = re.compile(combined, re.IGNORECASE)

    parts: list[str] = []
    last_end = 0
    for m in matcher.finditer(text):
        if not _word_boundary_ok(text, m.group(0), m.start()):
            continue
        token = tokens_by_group[m.lastindex - 1]
        parts.append(text[last_end : m.start()])
        parts.append(mapping[token])
        last_end = m.end()
    parts.append(text[last_end:])
    return "".join(parts)


def insert_placeholders(facts: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Substitute ``subject``/``object`` through the forward ``mapping``,
    leaving every other fact-dict field verbatim.

    One dict per input fact: ``subject``/``object`` go through
    :func:`_substitute_whole_words`; every other key (``predicate``,
    ``relation_type``, ``confidence``, ``speaker_id``, or anything else a
    caller's fact dict happens to carry) is copied through unchanged via
    ``{**f, ...}`` — the predicate is NEVER a substitution target, so a
    placeholder cannot be glued into it at this stage. A predicate shaped
    like ``language_proficiency_Language_3`` can still occur in the
    cloud's *returned* facts, which is why the deanon-stage predicate
    invariant in :func:`_apply_bindings` stays.

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
    fold-onto-token carve-in, which lets a model-authored
    ``{"RealName": "speaker0"}`` entry survive normalization for the CORE
    anonymizer table. This carve-in is deliberately NOT extended to
    ``placeholder_side="key"`` — the cloud ``bindings`` table — where a
    speaker-id-shaped KEY stays genuinely ambiguous (a cloud model has no
    authority to bind new content onto the identity anchor).

    ``placeholder_side="value"`` has no production caller: the local
    anonymizer never authors a ``{real: placeholder}`` mapping — SCAN only
    lists real values, and :func:`build_forward_table` mints every
    placeholder in code, including the fold-onto-``speaker{N}`` decision
    (from the ANCHOR call's ``anchor_names``). This variant (and its
    speaker-id carve-in) is retained because the ``bindings`` table
    (``placeholder_side="key"``) variant below shares this same normalizer,
    and that variant does have callers.

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
    has NO production caller — see the note above. The ``bindings`` table
    variant (``placeholder_side="key"``) has more than one caller
    (:meth:`~paramem.cloud.deanonymize.CloudScope.response`, and the
    unrelated per-session delta parser
    :func:`~paramem.graph.extractor._parse_enrichment_delta`) and its
    ``stats`` is not surfaced to a diagnostic by either.
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
            # placeholder-shaped and is dropped as ambiguous below — this
            # carve-in is what lets the CORE table builder actually receive
            # the entry (see the docstring note above on the ``value``
            # direction's caller).
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


def _rendering_fold(token: str) -> str:
    """The rendering-equivalence key for *token*: two declared tokens equal
    under this fold are one token.

    ``token.casefold()`` when *token* decomposes as a placeholder
    (:func:`_decompose_token`), *token* itself otherwise — a token with no
    rendering has no equivalence class beyond itself, so its comparisons
    stay exact-string.

    Used at both membership sites that enforce or diagnose the declared
    vocabulary's distinctness under the equivalence: :func:`_resolution_map`
    (CORE-LAST enforcement — a cloud binding whose key is rendering-equal to
    ANY core token, shown or not, or to another cloud binding, is inert)
    and :func:`_binding_collisions` (the diagnostic that names such
    bindings), each against a set of folds built once per call.
    """
    return token.casefold() if _decompose_token(token) is not None else token


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

    ``observed`` is ``None`` -> **CORE UNSCOPED** — every deanon call
    that has no cloud-observed scope to pass (e.g. a
    call outside the cloud-enrichment cycle, or a unit test exercising
    :func:`_apply_bindings`/:func:`_resolution_map` directly): every
    ``reverse`` entry is legal, ``cloud_bindings`` is merged in
    underneath it. An empty-set default here instead of ``None``
    would scope CORE to nothing and drop every fact on those paths — see
    the callers' ``observed: set[str] | None = None`` declarations.

    ``observed`` is a ``set`` -> **CORE SCOPED** to it: a ``reverse``
    entry resolves only when cloud was actually shown its placeholder, or
    a rendering of it (``key`` rendering-equal to a member of ``observed``
    — a token in the rendered facts cloud saw, or in the anonymized
    transcript, under :func:`_rendering_fold`'s equivalence). A
    ``cloud_bindings`` entry is admitted only when its key is
    rendering-equal to NO ``reverse`` key at all — shown or not — and to
    no OTHER ``cloud_bindings`` key; every member of such a collision
    class is inert (this is the declared-vocabulary distinctness
    invariant, enforced here rather than merely diagnosed — a rendering
    the external service writes back cannot itself carry the shown/unshown
    distinction, so a mint colliding with an unshown core token is exactly
    as unresolvable as one colliding with a shown one). A key
    rendering-equal to a member of ``observed`` is additionally a CONFLICT
    — surfaced separately, purely informationally, by
    :func:`_binding_collisions`. ``observed`` and every key compared
    against it hold exact tokens; :func:`_rendering_fold` is applied once
    per key, against sets/maps built once per branch.

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
        folded_observed = {_rendering_fold(tok) for tok in observed}
        folded_core = {_rendering_fold(k) for k in reverse if isinstance(k, str) and k}
        binding_folds = {k: _rendering_fold(k) for k in cloud_bindings if isinstance(k, str) and k}
        binding_fold_counts: dict[str, int] = {}
        for fold in binding_folds.values():
            binding_fold_counts[fold] = binding_fold_counts.get(fold, 0) + 1
        resolved.update(
            (k, v)
            for k, v in cloud_bindings.items()
            if isinstance(v, str)
            and v
            and k in binding_folds
            and binding_fold_counts[binding_folds[k]] == 1
            and binding_folds[k] not in folded_core
        )
        resolved.update(
            (k, v)
            for k, v in reverse.items()
            if isinstance(k, str)
            and isinstance(v, str)
            and k
            and v
            and _rendering_fold(k) in folded_observed
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
) -> list[str]:
    """Collision scan: names every ``cloud_bindings`` key that
    :func:`_resolution_map` makes inert — the diagnostic mirror of that
    function's admission rule, ALWAYS informational, never a rejection
    signal. A binding for a token cloud was SHOWN is inert under
    CORE-LAST precedence (:func:`_resolution_map` always resolves such a
    key to the CORE value); see that function's docstring for why a
    collision cannot corrupt resolution.

    A ``cloud_bindings`` key is named when it is rendering-equal
    (:func:`_rendering_fold`) to ANY ``reverse_mapping`` key — shown or
    not — or to another ``cloud_bindings`` key (every member of such a
    sibling collision class is named).

    Returns the sorted list of colliding keys, ``[]`` when the scan found
    nothing or ``cloud_bindings`` is empty/``None``.  Writes NOTHING to
    any caller-owned object: the ONE production caller,
    :func:`~paramem.cloud.deanonymize.deanonymize_facts`, writes the
    result onto its own ``DeanonResult.collisions``.
    """
    if not cloud_bindings:
        return []
    folded_core = {_rendering_fold(k) for k in reverse_mapping if isinstance(k, str) and k}
    binding_folds = {k: _rendering_fold(k) for k in cloud_bindings if isinstance(k, str) and k}
    binding_fold_counts: dict[str, int] = {}
    for fold in binding_folds.values():
        binding_fold_counts[fold] = binding_fold_counts.get(fold, 0) + 1
    collisions = sorted(
        k
        for k, fold in binding_folds.items()
        if fold in folded_core or binding_fold_counts[fold] > 1
    )
    if collisions:
        logger.warning(
            "cloud binding collision: %d placeholder(s) rendering-equal to "
            "the core reverse map or to a sibling cloud binding, inert "
            "under CORE-LAST precedence: %s.",
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
    here, whether bare or braced; only DECLARED-ness — membership in one
    of the two mapping tables — matters, so this helper is independent of
    the token shape in use.
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
    :func:`~paramem.cloud.anonymize.anonymize`, which inverts
    :func:`build_forward_table`'s own ``forward`` (already reconciled,
    pruned, and minted by the time it is returned) into ``reverse`` —
    :func:`build_forward_table` itself does not call this function; see
    its own docstring's Returns section.
    """
    out: dict[str, str] = {}
    for k, v in mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        out.setdefault(v, k)
    return out


def _index_identity_domain(
    identity_domain: Iterable[str] | None,
) -> tuple[dict[str, str], set[str]]:
    """Build the canonical-form -> domain-surface index for identity
    reconciliation ONCE per :func:`build_forward_table` call.

    ``identity_domain`` is the graph tier's ``chunk_nodes``, up to
    ``max_entities_per_pass`` — typically 50 entries. Returns
    ``(canon_to_domain, ambiguous_canon)``: a domain entry whose canonical
    form collides with an earlier one is a genuine ambiguity, recorded in
    ``ambiguous_canon`` rather than silently picking one.

    ``identity_domain is None`` (no domain to reconcile against — the
    session tier / chat egress / calibration) returns two empty
    collections; :func:`build_forward_table` skips its reconcile pass
    entirely in that case, so this function only ever produces the empty
    pair there.
    """
    canon_to_domain: dict[str, str] = {}
    ambiguous_canon: set[str] = set()
    if identity_domain is None:
        return canon_to_domain, ambiguous_canon
    for d in identity_domain:
        c = canonical(str(d))
        if c in canon_to_domain and canon_to_domain[c] != d:
            # Two distinct domain entries canonicalizing identically —
            # fail closed on this entry rather than silently pick one.
            ambiguous_canon.add(c)
        else:
            canon_to_domain[c] = d
    return canon_to_domain, ambiguous_canon


def _reconcile_to_domain(
    mapping: dict[str, _V],
    canon_to_domain: dict[str, str],
    ambiguous_canon: set[str],
) -> tuple[dict[str, _V], int]:
    """Re-key ``mapping`` onto ``canon_to_domain``'s domain surfaces,
    preserving each value verbatim.

    Every key is folded through :func:`~paramem.utils.identity.canonical`
    and matched against ``canon_to_domain``. A key whose canonical form is
    ambiguous, or has no domain match, is dropped and counted. Returns
    ``(reconciled_mapping, dropped_count)``.

    Value-generic (``_V``): this function copies the value verbatim and
    never inspects it, so :func:`build_forward_table` passes
    ``dict[str, _Group]`` (re-keying group membership by surface, before
    any placeholder exists) while its other callers pass ``dict[str,
    str]`` unchanged.
    """
    reconciled: dict[str, _V] = {}
    dropped = 0
    for real, value in mapping.items():
        c = canonical(real)
        if c in ambiguous_canon or c not in canon_to_domain:
            dropped += 1
            continue
        reconciled[canon_to_domain[c]] = value
    return reconciled, dropped


def _dropped_inert_entry(real: str, category: str) -> dict:
    """Build one ``scan_dropped_entries`` record for a forward-table key
    :func:`build_forward_table`'s prune pass drops: ``side="table"``,
    ``reason="inert"`` — distinguishing a key dropped because it never
    substituted anywhere in the payload from a scan-time drop
    (``side="scan"``, built by
    :func:`~paramem.cloud.anonymize_steps._dropped_scan_entry`).

    ``category`` is the owning group's own mint prefix, passed in by the
    caller — ``""`` for the speaker group, whose members carry no mint
    prefix at all. The same truncation cap
    (:data:`_MAX_MAPPING_TEXT_CHARS`) the scan path already uses for
    ``text``.
    """
    return {
        "category": category,
        "side": "table",
        "text": real[:_MAX_MAPPING_TEXT_CHARS],
        "reason": "inert",
    }


@dataclass(frozen=True)
class ForwardTable:
    """The resolved real -> placeholder table :func:`build_forward_table`
    returns, plus the two drop records the caller reports.

    Attributes:
        forward: The ``{real_name: placeholder}`` mapping that feeds
            :func:`insert_placeholders` — already reconciled (when a
            domain was given), pruned, and minted.
        rekey_dropped: Count of keys dropped by identity reconciliation
            (no domain match, or an ambiguous one) — ``0`` when
            ``identity_domain`` was ``None``.
        inert_entries: One :func:`_dropped_inert_entry` record per pruned
            key — a key that substitutes nowhere in ``tag_text``. The
            caller derives its own count from ``len(inert_entries)``: a
            count that can disagree with the list it summarises is not a
            field.
    """

    forward: dict[str, str]
    rekey_dropped: int
    inert_entries: tuple[dict, ...]


@dataclass(eq=False)
class _Group:
    """One set of real surfaces sharing a single placeholder (or the
    speaker token) — the unit of minting inside
    :func:`build_forward_table`. Module-private, never exported.

    ``eq=False`` keeps identity comparison/hashing (``is``, ``id()``): two
    groups are the same group only by object identity, never by
    coincidentally equal field values, so a group can be a ``set``/``dict``
    key (the containment pass's container-group pooling) without a
    field-value collision merging two distinct groups.

    Attributes:
        order: First-occurrence ordinal (the group's creation index) —
            ``min()``'d onto the survivor on a containment merge, so a
            merged group always emits at its EARLIEST member's position.
        owner: Index into ``scans`` — the one category whose containment
            pass may judge and re-point this group. Fixed at creation;
            never reassigned by a merge.
        prefix: The mint prefix — the owning category's
            ``ScrubCategory.prefix``, or ``""`` for the speaker group
            (which mints nothing).
        speaker: Whether this group's target is ``speaker_id`` — closed
            both ways (never judged, never a merge target); see
            :func:`build_forward_table`'s docstring.
        members: Surfaces, in first-occurrence (join) order.
    """

    order: int
    owner: int
    prefix: str
    speaker: bool
    members: list[str]


def _is_speaker_surface(value: str, *, anchor_names: frozenset[str], enrolled: str | None) -> bool:
    """The speaker-fold predicate: does *value* belong on the speaker
    group?

    ``enrolled is None`` (no display name to compare against — an
    anonymous-voice speaker) — attestation alone decides: *value* folds
    iff it is in *anchor_names*.

    ``enrolled`` set — consistency is SYMMETRIC: *value* folds when it is
    case-insensitively EQUAL to *enrolled* (no attestation needed — the
    enrolled name is trusted on its own), or when it is ATTESTED (in
    *anchor_names*) AND consistent with *enrolled* — either string starts
    with the other followed by a space (``"Alex Rivera"`` under enrolled
    ``"Alex"``, or ``"Alex"`` under enrolled ``"Alex Rivera"``). An
    attested surface that is merely a NAMESAKE (inconsistent with
    *enrolled*, e.g. ``"Mira"`` under enrolled ``"Alex"``) is refused — it
    mints an ordinary placeholder instead.
    """
    if enrolled is None:
        return value in anchor_names
    low, name = value.lower(), enrolled.lower()
    if low == name:
        return True
    return value in anchor_names and (low.startswith(name + " ") or name.startswith(low + " "))


def _open_group(groups: list[_Group], owner: int, prefix: str, *, speaker: bool) -> _Group:
    """Create, register, and return a fresh :class:`_Group` — used by
    :func:`build_forward_table`'s resolve pass. ``order`` is the group's
    creation index in *groups* (its first-occurrence ordinal).
    """
    group = _Group(order=len(groups), owner=owner, prefix=prefix, speaker=speaker, members=[])
    groups.append(group)
    return group


def _join(group: _Group, value: str, group_of: dict[str, _Group]) -> None:
    """Add *value* to *group* and register the membership in *group_of* —
    used by :func:`build_forward_table`'s resolve pass.
    """
    group.members.append(value)
    group_of[value] = group


def build_forward_table(
    scans: "Sequence[ScanResult]",
    *,
    tag_text: str,
    anchor_names: frozenset[str],
    speaker_id: str | None,
    speaker_name: str | None,
    identity_domain: Iterable[str] | None,
) -> ForwardTable:
    """Assemble the real -> placeholder forward table from verified SCAN
    results — code mints every placeholder; there is no model-authored
    mapping to normalize or invert (see the module docstring's "Model-free"
    note and :mod:`paramem.cloud.anonymize_steps`).

    THE one table constructor for the local-anonymizer CORE map: it
    resolves every scanned surface to a GROUP, reconciles group members
    onto ``identity_domain`` (when given), prunes members that substitute
    nowhere in ``tag_text``, and mints exactly one placeholder per
    surviving non-speaker group. Five passes, always in this order:

    0. **Gates.** The person-name category, the speaker fold TARGET, and
       the ENROLLED display name are each resolved once, before anything
       else. ``person_idx`` is the index into *scans* of the category
       whose prefix is the person prefix
       (:func:`~paramem.config.taxonomy.entity_type_to_prefix` ("person")),
       or ``-1`` when no such category is among the active scan
       categories — the operator narrowed ``sanitization.scrub`` away
       from person names. ``target`` is *speaker_id* itself, but only
       when it is truthy, well-shaped
       (:func:`~paramem.utils.identity.is_speaker_id`), AND a person
       category is active; otherwise ``None`` — and every later read of
       ``enrolled`` sits inside a ``target is not None`` branch, so
       ``target is None`` means the speaker's own name is never entered
       and never folds, whatever the caller passed. ``enrolled`` is
       *speaker_name* itself, but only when it is truthy, NOT itself
       speaker-id-shaped, AND a person category is active; otherwise
       ``None`` (an anonymous-voice speaker, or person-name scrubbing is
       off).
    1. **Resolve.** One sequential walk over *scans* (category order) x
       each scan's ``values`` (first-occurrence order) assigns every
       surface to a group: the speaker fold first
       (:func:`_is_speaker_surface`, only ever tested when ``target is
       not None``), else canonical-equality sharing onto an already-open
       group (:func:`~paramem.utils.identity.canonical` — SHARING, never
       deletion: every distinct verbatim surface still becomes its own
       forward-table key, sharing the group's eventual placeholder), else
       a fresh group owned by the current category (first-category-wins
       for a value later re-scanned under a second category, since the
       resolve walk never re-visits a value already in ``group_of``).
       After the walk, the enrolled name itself is entered as an
       additional forward key on the speaker group — but ONLY when
       ``target is not None`` and ``enrolled is not None`` and it is not
       already a member — so a name whose display form the tagger never
       separately tagged still egresses as the speaker token.
    2. **Containment**, per category, longest-first (see below).
    3. **Reconcile** — only when ``identity_domain is not None``: every
       surviving group member is re-keyed onto its domain surface via
       :func:`_index_identity_domain` / :func:`_reconcile_to_domain`; a
       member with no domain match, or an ambiguous one, is dropped and
       counted into ``rekey_dropped``.
    4. **Prune** — every member surviving reconciliation is tested against
       *tag_text* by :func:`_applied_whole_word_keys` (the one
       substitution walk); a member that substitutes nowhere is dropped
       and recorded as an inert entry (:func:`_dropped_inert_entry`)
       carrying its group's ``prefix`` as ``category`` (``""`` for the
       speaker group).
    5. **Mint & emit** — surviving groups are emitted in ascending
       first-occurrence order (``_Group.order``, ``min()``'d onto the
       survivor by every merge in pass 2, so a merged group emits at its
       earliest member's position); each non-speaker group takes exactly
       one :func:`mint_placeholder` call against the accumulating table —
       so a prefix's numbers run ``1..N`` with no holes, since nothing
       mints before pruning has removed every dead key. Within a group,
       members are emitted longest-canonical-form-first
       (:func:`~paramem.utils.identity.canonical` ``mode="spaces"``),
       stable — so :func:`invert_forward_mapping`'s first-key-wins rule
       always names the group's outermost (longest) surface, never a
       fragment.

    **Containment** (pass 2), per category *idx*: that category's
    non-empty, non-speaker-id string values are its *surfaces*; each
    surface's case/diacritic-preserving canonical form
    (:func:`~paramem.utils.identity.canonical` ``mode="spaces"``) is
    ``canon_forms``. The groups this category may JUDGE are exactly the
    groups whose ``owner == idx`` (excluding the speaker group, which is
    never judged) — every such group has at least one member in
    *surfaces* by construction (its founding member is what opened it
    during this category's resolve walk). Judged groups are sorted by the
    length of their longest member present in *surfaces*, descending,
    stable on ties (creation order) — a group is judged exactly once, by
    construction, never split across two containment outcomes. For each
    judged group, in that order: pool :func:`_whole_word_contains` matches
    across every one of the group's OWN-category members against every
    OTHER surface of the category whose group differs, and collect the
    distinct container GROUPS those matched surfaces belong to (a group
    identity, not a placeholder — nothing has minted yet). A speaker-group
    surface counts as a container for this ambiguity test exactly like any
    other. Exactly one container group, and it is NOT the speaker group,
    merges the judged group into it (the container keeps its identity,
    absorbs every member, and takes ``order = min(container.order,
    absorbed.order)``); two or more distinct containers, or a sole
    container that IS the speaker group, leaves the judged group with its
    own identity. Because judging runs longest-first, a container's own
    merge is already settled by the time a shorter group is judged, so a
    transitive chain (``"Ann"`` inside ``"Ann Marie"`` inside ``"Ann Marie
    Bell"``) collapses onto one group regardless of scan order.

    **The speaker group is CLOSED on both sides.** It is opened (once,
    lazily, on the first surface or the enrolled-name entry that needs it)
    with ``prefix=""`` and ``speaker=True`` — it mints nothing, and a
    pruned member of it therefore records ``category=""`` through the
    ordinary :func:`_dropped_inert_entry` path with no per-group carve-out.
    It is never judged (so containment can never merge it OUT) and never
    a merge target (so containment can never merge anything INTO it) — a
    scanned fragment of an attested or enrolled surface (``"Rivera"``
    under an attested ``"Alex Rivera"``) is an ordinary surface unless it
    is ITSELF attested or equal to the enrolled name: it mints its own
    placeholder. A surface reaches the speaker token only on evidence
    about that surface, never by containment inside another surface that
    had the evidence.

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
        tag_text: The complete marker-free outbound surface (history +
            transcript + fact lines) the prune pass (4) tests every
            surviving key against — the same text the caller later
            substitutes over via :func:`insert_placeholders` /
            :func:`_substitute_whole_words`.
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
        speaker_name: Runtime-known display name of the session's speaker,
            or ``None``. Consumed only when *speaker_id* is well-shaped AND
            a person-name category is active among *scans* — see the
            gates above (pass 0). A speaker-id-shaped value here
            (:func:`~paramem.utils.identity.is_speaker_id`) is never
            consumed either.
        identity_domain: The graph tier's own node list (or ``None`` on
            every other caller) — when given, pass 3 re-keys every
            surviving group member onto its domain surface.

    Returns:
        :class:`ForwardTable` — ``forward`` (the ``{real_name:
        placeholder}`` mapping :func:`insert_placeholders` consumes),
        ``rekey_dropped`` and ``inert_entries`` (the two drop records the
        caller reports; :func:`~paramem.cloud.anonymize.anonymize` derives
        ``reverse`` from ``forward`` itself, via
        :func:`invert_forward_mapping`, after dropping any entry whose
        VALUE is speaker-id-shaped).
    """
    person_prefix = entity_type_to_prefix("person")
    person_idx = next((i for i, s in enumerate(scans) if s.category.prefix == person_prefix), -1)
    target = speaker_id if (speaker_id and is_speaker_id(speaker_id) and person_idx >= 0) else None
    enrolled = (
        speaker_name
        if (speaker_name and not is_speaker_id(speaker_name) and person_idx >= 0)
        else None
    )

    groups: list[_Group] = []
    group_of: dict[str, _Group] = {}
    canon_index: dict[str, _Group] = {}
    speaker_group: _Group | None = None

    # 1 — resolve.
    for idx, scan in enumerate(scans):
        for value in scan.values:
            if not isinstance(value, str) or not value or is_speaker_id(value):
                continue
            if value in group_of:
                # Already assigned — a scan deduplicates within its own
                # category, but two categories can still list the
                # identical string (first-category-wins: it keeps the
                # group its first occurrence opened).
                continue
            if target is not None and _is_speaker_surface(
                value, anchor_names=anchor_names, enrolled=enrolled
            ):
                if speaker_group is None:
                    # prefix="" — the speaker group mints nothing, and a
                    # pruned member's inert record reads its category off
                    # this field.
                    speaker_group = _open_group(groups, idx, "", speaker=True)
                _join(speaker_group, value, group_of)
                canon_index.setdefault(canonical(value), speaker_group)
                continue
            shared = canon_index.get(canonical(value))
            if shared is not None:
                # Canonical SHARING, never deletion — see the docstring.
                _join(shared, value, group_of)
                continue
            fresh = _open_group(groups, idx, scan.category.prefix, speaker=False)
            _join(fresh, value, group_of)
            canon_index[canonical(value)] = fresh

    # 1b — the enrolled name as an additional key on the speaker group.
    # ``enrolled`` is None unless a person category is active, and every
    # read of it sits under ``target is not None``, so no branch exists
    # for a name without a well-shaped id.
    if target is not None and enrolled is not None and enrolled not in group_of:
        if speaker_group is None:
            speaker_group = _open_group(groups, person_idx, "", speaker=True)
        _join(speaker_group, enrolled, group_of)
        canon_index.setdefault(canonical(enrolled), speaker_group)

    # 2 — containment, per category, longest-first.
    for idx, scan in enumerate(scans):
        surfaces = [v for v in scan.values if isinstance(v, str) and v and not is_speaker_id(v)]
        surface_set = set(surfaces)
        canon_forms = {v: canonical(v, mode="spaces") for v in surfaces}

        judged_groups = [g for g in groups if g.owner == idx and g is not speaker_group]
        judged_groups.sort(
            key=lambda g: max(len(canon_forms[m]) for m in g.members if m in surface_set),
            reverse=True,
        )

        for group in judged_groups:
            own_members = [m for m in group.members if m in surface_set]
            containers: list[str] = []
            seen_containers: set[str] = set()
            for member in own_members:
                own_canon = canon_forms[member]
                for other in surfaces:
                    if other in seen_containers or group_of[other] is group:
                        continue
                    other_canon = canon_forms[other]
                    if len(other_canon) > len(own_canon) and _whole_word_contains(
                        other_canon, own_canon
                    ):
                        containers.append(other)
                        seen_containers.add(other)
            container_groups = {group_of[c] for c in containers}
            if len(container_groups) == 1:
                (container,) = container_groups
                if container is not speaker_group:
                    container.members.extend(group.members)
                    for m in group.members:
                        group_of[m] = container
                    container.order = min(container.order, group.order)
                    group.members = []

    # 3 — reconcile (skipped when identity_domain is None).
    member_to_group: dict[str, _Group] = {}
    for group in groups:
        for m in group.members:
            member_to_group[m] = group

    rekey_dropped = 0
    if identity_domain is not None:
        canon_to_domain, ambiguous_canon = _index_identity_domain(identity_domain)
        member_to_group, rekey_dropped = _reconcile_to_domain(
            member_to_group, canon_to_domain, ambiguous_canon
        )

    # 4 — prune: every surviving key is tested against tag_text.
    applied_keys = _applied_whole_word_keys(tag_text, member_to_group.keys())
    inert_entries: list[dict] = []
    surviving: dict[str, _Group] = {}
    for key, group in member_to_group.items():
        if key in applied_keys:
            surviving[key] = group
        else:
            inert_entries.append(_dropped_inert_entry(key, group.prefix))

    # 5 — mint & emit: ascending first-occurrence order, one mint per
    # surviving non-speaker group, members longest-canonical-form-first.
    # ``group_members`` is keyed by the ``_Group`` object itself — it is
    # ``eq=False`` (identity hash/eq) for exactly this: no parallel
    # ``id(group)`` bookkeeping needed to use it as a dict/set key.
    group_members: dict[_Group, list[str]] = {}
    for key, group in surviving.items():
        group_members.setdefault(group, []).append(key)

    ordered_groups = sorted(group_members, key=lambda g: g.order)

    forward: dict[str, str] = {}
    for group in ordered_groups:
        keys_sorted = sorted(
            group_members[group], key=lambda k: len(canonical(k, mode="spaces")), reverse=True
        )
        if group.speaker:
            # ``target`` (not ``speaker_id``) — it is the value that
            # actually gated this group's creation as a speaker group
            # (the gates pass), so it is non-None by construction here, unlike
            # ``speaker_id``'s own ``str | None`` signature type.
            placeholder = target
        else:
            placeholder = mint_placeholder(forward.values(), group.prefix)
        for key in keys_sorted:
            forward[key] = placeholder

    return ForwardTable(
        forward=forward,
        rekey_dropped=rekey_dropped,
        inert_entries=tuple(inert_entries),
    )


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
       (a) and (b) are pre-empted upstream by
       :func:`~paramem.graph.extractor._apply_enrichment_delta`, which
       drops an unresolvable ``add`` and reverts an unresolvable
       ``modify`` individually (per-triple, not a whole-delta rejection)
       BEFORE the fact ever reaches this function — so a bad mint sheds
       only the one action that carried it.  This sweep is the
       fail-closed backstop for whatever slips past that per-triple gate
       (in particular cause (c), and any fact this function is invoked on
       directly, outside the full session flow — e.g. its own unit
       tests). An anonymizer-stage leak is not a live cause:
       :func:`insert_placeholders` constructs the anon-stage fact
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

    ``resolution`` — when the caller already holds
    :attr:`~paramem.cloud.deanonymize.CloudScope.resolution` (the ONE
    production shape, via :func:`~paramem.cloud.deanonymize.
    deanonymize_facts`), pass it here instead of letting step 2 recompute
    ``_resolution_map(reverse, cloud_bindings, observed)`` a second time
    from the same three inputs. ``None`` (default) recomputes it for
    direct/unit-test callers with no ``CloudScope`` to hand.
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

    # Step 2 — substitute subject/object.
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
