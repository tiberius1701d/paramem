"""Indexed-key entry memory — production primitives.

Each entry has the canonical shape ``(key, subject, predicate, object)``.

Key properties:

* **One training example per fact.** Only the keyed-recall example is
  emitted — no natural-language second example, which halves per-cycle
  training time.
* **JSON envelope is** ``{"key", "subject", "predicate", "object"}`` —
  round-trip-clean, deterministic reconstruction.
* **Recall template is** ``"Recall the fact stored under key '{key}'."``
* **No natural-language recall path** — by design. The keyed prompt is the
  only guaranteed interface; natural-language questions are not trained.

SimHash fingerprinting constants and helpers are defined directly in this
module.  Registry-lifecycle helpers (``save_registry``, ``load_registry``)
are defined in :mod:`paramem.memory.persistence` and re-exported here so
callers can import them from either module.
"""

import hashlib
import json
import logging
from itertools import groupby

from paramem.training.dataset import SYSTEM_PROMPT, _tokenize_with_prompt_masking
from paramem.utils.identity import canonical

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape-agnostic SimHash constants and helpers
# (relocated from paramem.training.indexed_memory on 2026-05-20)
# ---------------------------------------------------------------------------

SIMHASH_BITS = 64
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def _tokenize_features(text: str) -> list[str]:
    """Tokenize text into word unigrams and bigrams for SimHash."""
    words = text.lower().split()
    features = list(words)
    for i in range(len(words) - 1):
        features.append(f"{words[i]} {words[i + 1]}")
    return features


def simhash_confidence(hash_a: int, hash_b: int, num_bits: int = SIMHASH_BITS) -> float:
    """Compute similarity confidence from two SimHash fingerprints.

    Returns ``1.0`` for identical fingerprints, ~``0.5`` for unrelated content.
    Based on normalized Hamming distance.

    Args:
        hash_a: First SimHash fingerprint integer.
        hash_b: Second SimHash fingerprint integer.
        num_bits: Number of bits in each fingerprint.

    Returns:
        Similarity score in ``[0.0, 1.0]``.
    """
    distance = bin(hash_a ^ hash_b).count("1")
    return 1.0 - (distance / num_bits)


def get_simhash(registry: dict, key: str) -> int | None:
    """Extract simhash from either simple or enriched registry format.

    Args:
        registry: Either ``{key: int}`` (simple) or ``{key: {"simhash": int, ...}}``
            (enriched).
        key: Registry key to look up.

    Returns:
        Integer SimHash fingerprint, or ``None`` when the key is absent.
    """
    entry = registry.get(key)
    if entry is None:
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, dict):
        return entry.get("simhash")
    return None


def _clean_generation_artifacts(text: str) -> str:
    """Strip markdown formatting artifacts from model output.

    Some instruct models (notably Gemma 2) inject ``**`` bold markers and
    excess newlines into generated JSON, breaking structure.  This cleans
    the text before JSON parsing.

    NOTE: Artifact patterns may evolve across reinforcement/consolidation
    cycles as the model retrains on its own output.  Monitor parse failure
    rates across cycles and extend patterns here if new artifacts emerge.

    Args:
        text: Raw model output string.

    Returns:
        Cleaned text with bold markers removed and excess newlines collapsed.
    """
    text = text.replace("**", "")
    # Collapse runs of 3+ newlines to one, preserving a 2-newline paragraph
    # break.  ``groupby`` yields maximal runs, which is what makes this exact:
    # repeated ``"\n\n\n" -> "\n"`` replacement is NOT equivalent (it leaves a
    # 4-newline run as two newlines).
    out: list[str] = []
    for ch, grp in groupby(text):
        run = len(list(grp))
        out.append("\n" if ch == "\n" and run >= 3 else ch * run)
    return "".join(out)


RECALL_TEMPLATE = "Recall the fact stored under key '{key}'."


# --- Key assignment ---


def assign_keys(
    triples: list[tuple[str, str, str]],
    start_index: int = 1,
    prefix: str = "graph",
) -> list[dict]:
    """Assign sequential ``<prefix><N>`` keys to a list of (subject, predicate, object) triples.

    Returns dicts with the four canonical fields used everywhere downstream.
    The ``start_index`` parameter lets callers concatenate key ranges
    (e.g. episodic keys 1–N, procedural keys N+1–M) without collisions.

    Args:
        triples: List of ``(subject, predicate, object)`` 3-tuples.
        start_index: First key index; the i-th triple gets key
            ``f"{prefix}{start_index + i}"``.
        prefix: Key prefix string.  Episodic/semantic keys use ``"graph"``
            (default); procedural keys use ``"proc"`` by convention.

    Returns:
        List of ``{"key", "subject", "predicate", "object"}`` dicts in the
        same order as the input.
    """
    return [
        {
            "key": f"{prefix}{start_index + i}",
            "subject": s,
            "predicate": p,
            "object": o,
        }
        for i, (s, p, o) in enumerate(triples)
    ]


# --- Training format ---


def _build_response(entry: dict) -> str:
    """Build the JSON response string for a single entry.

    Args:
        entry: Dict containing ``key``, ``subject``, ``predicate``, and ``object``.

    Returns:
        JSON string with exactly those four fields.
    """
    return json.dumps(
        {
            "key": entry["key"],
            "subject": entry["subject"],
            "predicate": entry["predicate"],
            "object": entry["object"],
        }
    )


def format_entry_training(
    entries: list[dict],
    tokenizer,
    max_length: int = 1024,
) -> list[dict]:
    """Build training examples — one keyed-recall example per entry.

    No natural-language second example. One example per entry halves
    the per-cycle training time.

    Args:
        entries: List of entry dicts (each with ``key``, ``subject``,
            ``predicate``, ``object``).
        tokenizer: HuggingFace tokenizer compatible with
            :func:`paramem.training.dataset._tokenize_with_prompt_masking`.
        max_length: Maximum token length per example (passed to the tokenizer).

    Returns:
        List of pre-tokenized training example dicts with ``input_ids``,
        ``attention_mask``, and ``labels`` (prompt tokens masked to -100).
    """
    from paramem.models.loader import adapt_messages

    examples = []
    for entry in entries:
        recall_prompt = RECALL_TEMPLATE.format(key=entry["key"])
        recall_response = _build_response(entry)
        messages = adapt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": recall_prompt},
                {"role": "assistant", "content": recall_response},
            ],
            tokenizer,
        )
        examples.append(_tokenize_with_prompt_masking(messages, tokenizer, max_length))
    return examples


# --- Recall parsing ---


def parse_recalled_entry(text: str) -> dict | None:
    """Parse a single recalled entry JSON object from model output.

    Tries the raw text first, then — only if cleaning changed anything — retries
    on :func:`paramem.memory.entry._clean_generation_artifacts`'d text
    (strips markdown bold markers and excess newlines that some instruct models
    inject).

    Uses progressive first-object extraction: scans forward for ``{``, tries each
    ``}`` (via ``raw_decode``, never ``rfind``) to avoid swallowing chained objects.

    A list-valued ``object`` field is coerced to a comma-joined string so
    multi-value objects survive the round-trip.

    Args:
        text: Raw model output string.

    Returns:
        ``{"key", "subject", "predicate", "object"}`` dict, or ``None`` if the
        output is not parseable or does not contain the required fields.
    """
    text = text.strip()
    required = {"key", "subject", "predicate", "object"}
    decoder = json.JSONDecoder()

    def _coerce(v) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    def _try_parse(t: str) -> dict | None:
        for i, ch in enumerate(t):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(t[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and required.issubset(obj.keys()):
                return {k: _coerce(obj[k]) for k in ("key", "subject", "predicate", "object")}
        return None

    # Try raw text first, then cleaned (only if cleaning changed anything).
    result = _try_parse(text)
    if result is not None:
        return result

    cleaned = _clean_generation_artifacts(text)
    if cleaned != text:
        return _try_parse(cleaned)

    return None


# --- SimHash fingerprinting ---


def compute_simhash(
    key: str,
    subject: str,
    predicate: str,
    obj: str,
    num_bits: int = SIMHASH_BITS,
) -> int:
    """Compute a SimHash fingerprint from key + subject + predicate + object.

    The fingerprint is over exactly the SPO values it is given, hashed
    verbatim — this function does no folding of its own. Identity
    canonicalization happens once upstream (at the GraphMerger node-identity
    boundary); its two callers, :func:`entry_simhash` (registration) and
    :func:`verify_confidence` (recall), additionally apply a symmetric
    **spaces-only** fold (:func:`~paramem.utils.identity.canonical` with
    ``mode="spaces"``) to each field before calling this function, so a
    ``_``↔space drift between what was registered and what a recall echoes
    back does not desync the fingerprint. The SimHash tokenizer lowercases, so
    case never reaches the fingerprint; what this narrower fold preserves over
    ``mode="full"`` is the diacritic/NFC distinction — ``"café"`` and
    ``"cafe"`` still hash differently, while only a ``_``/space difference
    (always the same fact) is merged. This is NOT the full ``canonical()`` fold
    that once ran here and was removed — that one also folded
    diacritics/whitespace, which desynced the fingerprint against a raw
    registry; the spaces-only fold is narrower and applied identically by both
    callers, so a fingerprint computed at registration still matches the one
    recomputed from a correct recall.

    Uses unigram+bigram feature tokenization and a bit-vote algorithm. The key is
    included so that identical triple content under different keys produces
    different fingerprints — catches hallucinations where the model echoes the
    queried key but returns another key's content.

    Args:
        key: The ``graphN`` key string.
        subject: Triple subject.
        predicate: Triple predicate.
        obj: Triple object.
        num_bits: Number of bits in the fingerprint (default 64).

    Returns:
        A ``num_bits``-bit integer fingerprint.
    """
    text = f"{key} {subject} {predicate} {obj}"
    features = _tokenize_features(text)

    if not features:
        return 0

    sums = [0] * num_bits
    for feature in features:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        for i in range(num_bits):
            if h & (1 << i):
                sums[i] += 1
            else:
                sums[i] -= 1

    fingerprint = 0
    for i in range(num_bits):
        if sums[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def entry_simhash(entry: dict) -> int:
    """Compute the SimHash fingerprint for an entry from its OWN stored fields.

    This is THE single primitive for turning an entry dict into a fingerprint.
    Every mint/registration site must route through this function instead of
    reconstructing a ``compute_simhash(key, subject, predicate, object)`` call
    inline against some other representation of the fact (e.g. a graph node
    key). :func:`verify_confidence` rebuilds its candidate fingerprint from
    exactly these same entry fields (``entry.get("subject")`` /
    ``entry.get("predicate")`` / ``entry.get("object")``) at recall time — so
    hashing anything other than the entry's own ``subject``/``predicate``/
    ``object`` desyncs the registered fingerprint from what recall verifies
    against, silently corrupting the confidence score and — below
    :data:`DEFAULT_CONFIDENCE_THRESHOLD` — dropping the fact.

    ``subject``/``predicate``/``object`` are each passed through
    :func:`~paramem.utils.identity.canonical` with ``mode="spaces"`` before
    hashing — a symmetric fold shared with :func:`verify_confidence` that
    merges only a ``_``↔space surface drift (case preserved) so recall of the
    same fact under either surface still verifies. ``key`` is hashed verbatim.

    Args:
        entry: Dict containing at minimum ``key``, ``subject``, ``predicate``,
            and ``object`` — the exact fields written into the store / used to
            build the training example.

    Returns:
        The 64-bit SimHash fingerprint for this entry.
    """
    return compute_simhash(
        entry["key"],
        canonical(entry["subject"], mode="spaces"),
        canonical(entry["predicate"], mode="spaces"),
        canonical(entry["object"], mode="spaces"),
    )


def verify_confidence(
    recalled: dict,
    registry: dict[str, int] | dict[str, dict] | None = None,
) -> float:
    """Verify a recalled entry against a SimHash registry.

    Contract:

    - Returns ``1.0`` when no registry is provided (no verification).
    - Returns ``0.0`` when the key is absent from the registry (untrained key).
    - Returns a ``0.0–1.0`` SimHash similarity score otherwise (higher = more
      likely genuine recall).

    Supports both the simple ``{key: int}`` registry shape and the enriched
    ``{key: {"simhash": int, ...}}`` shape via
    :func:`paramem.memory.entry.get_simhash`.

    ``subject``/``predicate``/``object`` are each passed through
    :func:`~paramem.utils.identity.canonical` with ``mode="spaces"`` before
    hashing — the same fold :func:`entry_simhash` applies at registration, so
    a recalled fact that echoes back a ``_``↔space surface drift (e.g. the
    trained predicate ``works_at`` recalled as ``"works at"``) still verifies
    at high confidence.

    Args:
        recalled: Dict containing at minimum ``key``, ``subject``,
            ``predicate``, and ``object``.
        registry: Optional SimHash registry mapping key → fingerprint.

    Returns:
        Confidence score in ``[0.0, 1.0]``.
    """
    if registry is None:
        return 1.0

    key = recalled.get("key", "")
    expected = get_simhash(registry, key)
    if expected is None:
        return 0.0

    actual = compute_simhash(
        key,
        canonical(recalled.get("subject", ""), mode="spaces"),
        canonical(recalled.get("predicate", ""), mode="spaces"),
        canonical(recalled.get("object", ""), mode="spaces"),
    )
    return simhash_confidence(actual, expected)


# --- Registry management ---


def build_registry(entries: list[dict]) -> dict[str, int]:
    """Build a SimHash registry from entries.

    Returns a mapping of ``key → 64-bit SimHash fingerprint``.
    This is the simple format used by the training pipeline.

    Args:
        entries: List of entry dicts, each containing ``key``,
            ``subject``, ``predicate``, and ``object``.

    Returns:
        Dict mapping each key to its SimHash fingerprint.
    """
    return {p["key"]: entry_simhash(p) for p in entries}


# --- Probe ---


def finalize_recalled(
    raw: str,
    key: str,
    registry: dict | None,
    confidence_threshold: float,
) -> dict:
    """Turn a raw model output into the recalled-entry contract dict.

    Used by the batched recall path in recall_eval.py. Returns either the
    parsed-entry dict augmented with confidence/raw_output/fact_text, or a
    failure-reason dict.
    """
    parsed = parse_recalled_entry(raw)
    if parsed is None:
        logger.debug("Parse failure for key '%s': %s", key, raw[:200])
        return {"raw_output": raw, "failure_reason": "parse_failure"}
    if parsed["key"] != key:
        logger.debug("Key mismatch: queried '%s', got '%s'", key, parsed["key"])
        return {"raw_output": raw, "failure_reason": f"key_mismatch:{parsed['key']}"}
    confidence = verify_confidence(parsed, registry)
    if confidence < confidence_threshold:
        logger.debug(
            "Low confidence for key '%s': %.3f < %.3f threshold",
            key,
            confidence,
            confidence_threshold,
        )
        return {"raw_output": raw, "failure_reason": f"low_confidence:{confidence:.3f}"}
    parsed["confidence"] = confidence
    parsed["raw_output"] = raw
    parsed["fact_text"] = entry_fact_text(parsed)
    return parsed


# --- Fact-text helper ---


def entry_fact_text(entry: dict) -> str:
    """Render a recalled entry as prose — the identity → display boundary.

    This is THE render boundary: it turns the *stored identity form* of a
    triple into human- and model-facing text.  Identity form and rendered form
    are different layers.  Identity is what gets stored, keyed, compared and
    trained; rendering is where a fact becomes prose.  Output reaches inference
    context, cloud prompts and TTS, so it must read as language.

    The two sides are treated differently because they are stored differently:

    * ``subject`` / ``object`` are **identity-folded** — the graph merger keys
      them via :func:`paramem.utils.identity.canonical` at the node-identity
      boundary and keeps the first-seen display surface in the node's
      ``attributes["name"]`` — so they are emitted as-is here; no further
      transformation happens at this boundary.  A ``speaker{N}`` token in
      subject or object position is emitted verbatim too — it is NOT resolved
      to a display name here.  Every model-facing surface (recalled facts,
      reasoning context, generated replies) stays in token space; a token is
      substituted for a human-readable name exactly once, at the reply
      boundary, by :func:`~paramem.server.speaker.resolve_speaker_tokens`,
      never at fact-render time.
    * ``predicate`` is stored in **identity form**, which is already
      space-form (``canonical``'s blank fold collapses ``_``/whitespace to a
      single space, e.g. ``"has sister-in-law"``), so it is used directly —
      no ``_``→space substitution happens here any more.  ``-`` is not a
      blank and was never touched.

    Used by inference consumers so string construction stays in the probe
    layer — callers read ``result["fact_text"]``.

    Args:
        entry: Dict containing at minimum ``subject``, ``predicate``, and
            ``object`` keys.

    Returns:
        Human-readable fact string, e.g. ``"Alex lives in Heilbronn"`` (from
        the stored predicate ``"lives in"``), or ``"speaker0 lives in
        Heilbronn"`` when the subject is still a raw token.
    """
    subject = entry["subject"]
    obj = entry["object"]
    predicate = " ".join(entry["predicate"].split())
    return f"{subject} {predicate} {obj}"


# --- Registry persistence (re-exported for convenience) ---

# These helpers are defined in paramem.memory.persistence (the registry I/O
# module) and re-exported here so callers can import from either location.
# Triple-hop re-export chain:
#   paramem.memory.persistence  ←defines←  save_registry / load_registry / …
#   paramem.memory.entry        ←re-exports← (this block)
#   paramem.memory.__init__     ←re-exports← (package surface)
# persistence.py does NOT import entry.py → no import-time cycle.

from paramem.memory.persistence import (  # noqa: E402, F401
    load_registry,
    save_registry,
)
