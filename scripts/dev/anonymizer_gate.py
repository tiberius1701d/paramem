"""The anonymizer gate: one tool, one corpus, one scorer, one verdict.

Runs the production anonymize chain (``paramem.graph.flows.anonymize_turn``
for a transcript-shaped entry, ``paramem.cloud.anonymize.anonymize`` for a
facts-shaped entry — the same call ``paramem.server.calibrate.
dispatch_anonymize_facts`` makes) over the unified fictional corpus at
``tests/fixtures/anonymizer_gate.json``, scores the result against that
corpus's gold spans, and prints a scorecard plus a regression verdict
against the last accepted baseline (``data/ha/calibration/
anonymizer_gate/baseline.json``). No threshold is hardcoded: a run either
matches or beats every primary column of the last accepted scorecard, or
every column it regresses on is named for the owner to read.

Usage::

    set -a && source .env && set +a && \\
      $HOME/miniforge3/envs/paramem/bin/python \\
      scripts/dev/anonymizer_gate.py [options]

A model-bearing run runs inside the experiment GPU guard (the server
released for the whole run) and restores the server's GPU with ``POST
<--server URL>/gpu/acquire`` in a ``finally``, retrying on a busy server
(503 while a fold runs, 409 during a base swap). Two modes never touch a
model or the GPU: ``--dry-run``, which assembles and validates the corpus
then loads the bare tokenizer on the CPU
(:func:`~paramem.models.loader.load_tokenizer`) to re-measure the two
anonymizer prompt skeletons (see below); and ``--resume`` onto a run
directory that is already complete — every corpus entry (after any
``--limit``) already has a written artifact on disk — which scores
straight from those artifacts instead of running anything, also loading
only the bare tokenizer for the same skeleton print. A ``--resume`` run
still missing entries, or any run without ``--resume``, proceeds under the
GPU guard as normal.

The corpus run chunks its GPU burst: every ``--cooldown-every`` (default
20) model-bearing entries, the tool pauses for the GPU to cool
(``wait_for_cooldown``) before continuing, the same chunked-burst pattern
any long GPU run in this project uses. ``--dry-run`` never runs a
model-bearing entry, so it never pauses.

Every run, ``--dry-run`` included, re-measures the SCAN and ANCHOR prompt
skeletons against their pinned reference constants
(``paramem.utils.tokens.ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS`` /
``ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS``): it renders each section with
an empty payload through the production section renderers
(``paramem.cloud.anonymize_steps.render_scan_section`` /
``render_anchor_section``) and the production chat template
(``paramem.cloud.anonymize_steps.render_call_prompt``), counts each
directly with the tokenizer (``paramem.utils.tokens.encode_rendered``,
never ``estimate_tokens``), and prints measured against reference for
both.

The category set scrubbed is never read from a live server config's
``sanitization.scrub`` — ``tests/fixtures/server.yaml`` deliberately empties
that field (``[]``) to keep every OTHER test that loads the fixture free of
a real SCAN call, so reading it here would opt every run out silently. The
default is the shipped default scrub set instead —
:class:`~paramem.server.config.SanitizationConfig` constructed with no
arguments, resolving its own dataclass default (``_DEFAULT_SCRUB``) into
``scrub_categories`` — overridable with ``--scrub <hint> [<hint> ...]``,
resolved through :func:`~paramem.config.taxonomy.resolve_scrub_categories`
exactly as the operator's own ``sanitization.scrub`` would be. The model
loaded is unaffected by this: it stays the fixture's own model entry
(``tests/fixtures/server.yaml``'s ``model:``) unless ``--model`` overrides it.

The scorer (``score_entry`` / ``score_corpus`` / ``Result`` below) is the
one implementation of the gate's scoring rules; ``tests/
test_anonymizer_gate_scorer.py`` imports it directly rather than
reimplementing it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from paramem.cloud.anonymize import AnonymizedContract, assemble_payload, render_fact_lines
from paramem.cloud.placeholders import applied_whole_word_keys, word_boundary_ok
from paramem.config.taxonomy import ScrubCategory, resolve_scrub_categories
from paramem.training.thermal_throttle import wait_for_cooldown
from paramem.utils.tokens import ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS, ANONYMIZE_SCAN_REPLY_RATIO
from paramem.utils.turn_markers import format_turn

# Mirrors the ``_REPO_ROOT`` pattern used by every other ``scripts/dev/*.py``
# calibration tool (e.g. ``calibrate_prompts.py``) — paths resolve the same
# regardless of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "anonymizer_gate.json"
_RUN_ROOT = _REPO_ROOT / "data" / "ha" / "calibration" / "anonymizer_gate"
_BASELINE_PATH = _RUN_ROOT / "baseline.json"

_SERVER_BASE_URL = "http://127.0.0.1:8420"

# The project's everyday GPU working threshold and a bounded wait, the same
# pair used at the other standalone-script cooldown call site
# (scripts/dev/probe_orphan_classification_live.py).
_COOLDOWN_THRESHOLD_C = 52
_COOLDOWN_MAX_WAIT_S = 600

# The "contact" primary column of the scorecard: phone, email and street
# address together (paramem's Phone/Email/Address scrub rows) — never
# Profile, which is a different kind of value entirely.
_CONTACT_PREFIXES = frozenset({"Phone", "Email", "Address"})


def resolve_categories(scrub_hints: list[str] | None) -> tuple[ScrubCategory, ...]:
    """Resolve the categories this run scrubs.

    *scrub_hints* (the ``--scrub`` CLI override) resolved through
    :func:`~paramem.config.taxonomy.resolve_scrub_categories` when given.
    Otherwise the shipped default scrub set:
    :class:`~paramem.server.config.SanitizationConfig` constructed with no
    arguments, whose own dataclass default (``_DEFAULT_SCRUB``) its
    ``__post_init__`` resolves into ``scrub_categories`` — never a live
    server config's ``sanitization.scrub``, which ``tests/fixtures/
    server.yaml`` deliberately empties (``[]``) to keep every other test
    that loads that fixture free of a real SCAN call.

    Args:
        scrub_hints: ``--scrub`` values, or ``None`` for the shipped default.

    Returns:
        Active rows in ``anonymizer.scrub`` list order.
    """
    if scrub_hints is not None:
        return resolve_scrub_categories(scrub_hints)
    from paramem.server.config import SanitizationConfig

    return SanitizationConfig().scrub_categories


# ---------------------------------------------------------------------------
# Corpus loading and lightweight structural validation.
# ---------------------------------------------------------------------------


def load_corpus(path: Path = _FIXTURE_PATH) -> list[dict]:
    """Load the unified anonymizer-gate corpus.

    Args:
        path: Override path, for tests that build their own fixture copy.

    Returns:
        The corpus's ``entries`` list, verbatim.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["entries"]


def entry_surfaces(entry: dict) -> list[tuple[object, str]]:
    """Every text surface *entry* offers gold positions into.

    A facts-shaped entry (``"facts"`` present) has exactly one surface,
    keyed ``"facts"``: the rendered fact-lines text
    (:func:`~paramem.cloud.anonymize.render_fact_lines`) — the same text
    the SCAN call reads for a facts-shaped ``anonymize()`` call. A
    transcript-shaped entry has one surface per ``history`` turn (keyed by
    its index) plus the entry's own current turn (keyed ``-1``).

    Returns:
        ``[(turn_id, text), ...]`` in the order gold entries reference —
        history turns first, then ``-1`` (or the single ``"facts"`` entry).
    """
    if "facts" in entry:
        return [("facts", render_fact_lines(entry["facts"]))]
    surfaces = [(i, t["text"]) for i, t in enumerate(entry.get("history", ()))]
    surfaces.append((-1, entry["text"]))
    return surfaces


def validate_corpus(entries: list[dict]) -> None:
    """Mechanical, universal corpus checks — schema shape, id uniqueness,
    and gold-offset correctness against the entry's own text.

    Author-specific guarantees (forbidden real names, the honorific rule,
    word-count bounds by kind, non-overlapping spans, decoys occurring in
    the text) live in ``tests/test_anonymizer_gate_corpus.py`` instead —
    this is the subset ``--dry-run`` can cheaply re-check on every
    invocation without duplicating the test's authoring-time assertions.

    Raises:
        ValueError: The corpus fails any check, naming the entry and cause.
    """
    seen_ids: set[str] = set()
    for entry in entries:
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"entry with missing/invalid id: {entry!r}")
        if entry_id in seen_ids:
            raise ValueError(f"duplicate id: {entry_id!r}")
        seen_ids.add(entry_id)

        has_text = "text" in entry
        has_facts = "facts" in entry
        if has_text == has_facts:
            raise ValueError(
                f"{entry_id}: exactly one of 'text'/'facts' is required, got "
                f"text={has_text} facts={has_facts}"
            )
        for key in ("lang", "casing", "kind", "description", "gold", "decoys"):
            if key not in entry:
                raise ValueError(f"{entry_id}: missing required key {key!r}")

        surfaces_by_id = dict(entry_surfaces(entry))
        for gold in entry["gold"]:
            for key in ("value", "category", "turn", "start", "end"):
                if key not in gold:
                    raise ValueError(f"{entry_id}: gold entry missing {key!r}: {gold!r}")
            text = surfaces_by_id.get(gold["turn"])
            if text is None:
                raise ValueError(f"{entry_id}: gold references unknown turn {gold['turn']!r}")
            if text[gold["start"] : gold["end"]] != gold["value"]:
                raise ValueError(
                    f"{entry_id}: gold offset mismatch for {gold['value']!r} "
                    f"(turn {gold['turn']!r}, {gold['start']}:{gold['end']} "
                    f"-> {text[gold['start'] : gold['end']]!r})"
                )


# ---------------------------------------------------------------------------
# The scorer: one implementation, imported by tests/test_anonymizer_gate_scorer.py.
# ---------------------------------------------------------------------------


def find_occurrences(text: str, values: list[str]) -> list[tuple[str, int, int]]:
    """Replicate the production longest-first, edge-aware-boundary walk
    (:func:`~paramem.cloud.placeholders._substitute_whole_words_and_applied`)
    to find WHERE each of *values* occurs in *text*: longest value first at
    each position, left to right, each character of *text* consumed by at
    most one match, a candidate position accepted only when the text equals
    the value there AND :func:`~paramem.cloud.placeholders.word_boundary_ok`
    holds at that position.

    Unlike :func:`~paramem.cloud.placeholders.applied_whole_word_keys`
    (which only reports WHICH keys matched), this also reports WHERE — the
    positional information the scorer needs to test a match against a
    gold span. Never used as the self-check's own reference (see
    :func:`_self_check`, which uses ``applied_whole_word_keys`` directly).

    *values* is ONE population — the forward table's keys, the reverted
    surfaces, or the unknown-word values — never a mixture of them.
    Production substitutes the forward table by itself
    (:func:`~paramem.cloud.placeholders._substitute_whole_words`, over
    ``table.forward``), so the forward keys' placements here are the text
    that actually gets replaced; the other two populations are values
    production substitutes nowhere, located only so the gate can test
    them against a gold span. Each character of *text* is consumed by at
    most one match, so two populations walked together would compete for
    characters: the longer surface would take the position and the other
    population's own match there would go unplaced.

    Returns:
        Matches found, as ``(value, start, end)``, in left-to-right order.
    """
    keys_sorted = sorted({v for v in values if isinstance(v, str) and v}, key=len, reverse=True)
    occurrences: list[tuple[str, int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        matched = False
        for key in keys_sorted:
            end = pos + len(key)
            if end > n or text[pos:end] != key:
                continue
            if not word_boundary_ok(text, key, pos):
                continue
            occurrences.append((key, pos, end))
            pos = end
            matched = True
            break
        if not matched:
            pos += 1
    return occurrences


def _occurrences_over_surfaces(
    surfaces: list[tuple[object, str]], values: list[str]
) -> list[tuple[object, str, int, int]]:
    """Walk ONE population of *values* over each of *surfaces* in turn
    (:func:`find_occurrences` — see there for the one-population rule).

    Args:
        surfaces: ``[(turn_id, text), ...]`` as :func:`entry_surfaces`
            returns them; gold offsets are relative to one surface's own
            text, so each is walked on its own.
        values: The one population to locate.

    Returns:
        ``[(turn_id, value, start, end), ...]``, surface by surface in
        *surfaces* order and left to right within each surface.
    """
    return [
        (turn_id, value, start, end)
        for turn_id, text in surfaces
        for value, start, end in find_occurrences(text, values)
    ]


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Whether ``[a_start, a_end)`` and ``[b_start, b_end)`` share any character."""
    return max(a_start, b_start) < min(a_end, b_end)


def span_covers(outer_start: int, outer_end: int, inner_start: int, inner_end: int) -> bool:
    """Whether ``[outer_start, outer_end)`` contains ``[inner_start, inner_end)`` in full."""
    return outer_start <= inner_start and outer_end >= inner_end


def _entry_tag_text(entry: dict) -> str:
    """Reconstruct the exact concatenated payload text
    (:attr:`~paramem.cloud.anonymize.TagPayload.tag_text`) the SCAN call
    actually read for *entry* — production's own assembly
    (:func:`~paramem.cloud.anonymize.assemble_payload`), fed the same
    ``history``/``text``/``facts`` the entry's own anonymize call used.

    Used only by :func:`_self_check` — never for occurrence search, which
    is per-surface (see :func:`entry_surfaces`) since gold offsets are
    relative to one turn's own text, not the concatenated payload.
    """
    if "facts" in entry:
        return assemble_payload([], "", entry["facts"]).tag_text
    history_lines = [format_turn(t["role"], t["text"]) for t in entry.get("history", ())]
    transcript = format_turn("user", entry["text"])
    return assemble_payload(history_lines, transcript, []).tag_text


def _self_check(entry: dict, forward: dict[str, str], scrubbed_values: set[str]) -> None:
    """Check the scorer's own occurrence walk, and the forward table's own
    prune guarantee, against production's own substitution walk
    (:func:`~paramem.cloud.placeholders.applied_whole_word_keys`) over the
    reconstructed payload text. Two independent arms, each raised with its
    own message so the cause is unambiguous:

    1. **Walk agreement.** *scrubbed_values* — the distinct ``forward``
       values the scorer's own occurrence walk actually placed somewhere
       in the entry's surfaces (:func:`find_occurrences`, run per surface
       via :func:`entry_surfaces` — see :func:`score_entry`) — must equal
       what production's own :func:`~paramem.cloud.placeholders.
       applied_whole_word_keys` reports over the reconstructed
       CONCATENATED payload text (:func:`_entry_tag_text`). A mismatch
       means the scorer's per-surface reconstruction of what substitutes
       disagrees with what production's own walk over the payload it
       actually built would find — never a normal scoring outcome.
    2. **Prune guarantee.** Every ``forward`` key is guaranteed
       substitutable somewhere in the payload by production's own
       inert-key prune (:func:`~paramem.cloud.placeholders.
       build_forward_table`) before the contract is ever returned — a key
       present in ``forward`` that occurs nowhere in the reconstructed
       payload text is therefore a genuine disagreement between this
       scorer's reconstruction and what actually produced the contract (a
       stale entry, an edited corpus after the contract was captured, or a
       defect in this module), never a normal scoring outcome.

    Args:
        entry: The corpus entry being scored.
        forward: The contract's own forward table.
        scrubbed_values: The distinct ``forward`` values the scorer's own
            occurrence walk placed somewhere in *entry*'s surfaces.

    Raises:
        AssertionError: Either arm disagrees, naming the entry id and both
            sets it compared.
    """
    tag_text = _entry_tag_text(entry)
    applied = applied_whole_word_keys(tag_text, forward.keys())

    if scrubbed_values != applied:
        raise AssertionError(
            f"self-check disagreement on entry {entry['id']!r}: the scorer's own "
            f"occurrence walk placed {sorted(scrubbed_values)!r}, but production's "
            f"own applied_whole_word_keys reports {sorted(applied)!r} over the "
            "reconstructed payload text"
        )

    missing = set(forward) - applied
    if missing:
        raise AssertionError(
            f"self-check disagreement on entry {entry['id']!r}: production's own "
            f"forward table names {sorted(missing)!r} as substitutable, but they "
            "do not occur (whole-word) anywhere in the reconstructed payload text"
        )


def _unknown_word_values(contract: AnonymizedContract) -> list[str]:
    """The real values the SCAN call dropped as ``reason="unknown_word"``.

    The drop record itself now carries the value directly on ``text``
    (:func:`~paramem.cloud.anonymize_steps._dropped_scan_entry` — ``word``
    holds the model's own unrecognised keyword instead), so this is a
    plain projection, never a re-parse of ``contract.raw``.
    """
    return [d["text"] for d in contract.scan_dropped_entries if d["reason"] == "unknown_word"]


@dataclass
class Result:
    """One run's scored aggregates and full detail lists."""

    name: str = "run"
    tot: Counter = field(default_factory=Counter)
    by_lang: dict = field(default_factory=lambda: defaultdict(Counter))
    by_casing: dict = field(default_factory=lambda: defaultdict(Counter))
    miss_list: list = field(default_factory=list)
    partial_list: list = field(default_factory=list)
    junk_list: list = field(default_factory=list)
    wrong_type_list: list = field(default_factory=list)
    unsub_list: list = field(default_factory=list)
    decoys_scrubbed: list = field(default_factory=list)
    unknown_word_census: Counter = field(default_factory=Counter)
    unknown_word_by_word: Counter = field(default_factory=Counter)
    skipped: list = field(default_factory=list)
    skipped_gold_names: int = 0
    skipped_gold_contact: int = 0
    largest_scan_reply_ratio: float | None = None
    largest_scan_reply_ratio_entry: str | None = None
    scan_reply_run_output_tokens: int = 0
    scan_reply_run_payload_tokens: int = 0
    scan_call_run_total: int = 0
    largest_scan_call_count: int | None = None
    largest_scan_call_count_entry: str | None = None


def score_entry(
    entry: dict, contract: AnonymizedContract, configured: set[str], result: Result
) -> None:
    """Score one entry's completed :class:`AnonymizedContract` into *result*.

    Implements the gate's own matching rules over the forward-table
    contract: scrubbed values are ``contract.forward``'s keys; a reverted
    value's real surface and out-of-scope row name come from its
    ``reason="reverted"`` drop record; the unsubstitutable count is
    ``contract.inert_dropped``; the unknown-word census by gold category
    uses each ``reason="unknown_word"`` drop record's own ``text`` (the
    real value, :func:`_unknown_word_values`); the unknown-word census by
    the model's own malformed word uses that same record's ``word``.

    Each of those three populations — the forward keys, the reverted
    surfaces, the unknown-word values — is walked over the entry's
    surfaces by itself (:func:`_occurrences_over_surfaces`), so a forward
    key's placements are exactly the text production's own substitution
    over the forward table replaces, and a reverted or unknown-word
    surface is located wherever it sits, including inside or across one
    of those replacements.

    Also updates the run's scan-reply-ratio tracking: for every
    ``contract.call_tokens`` record labelled ``"anonymize.scan"``, the
    payload tokens are that call's ``prompt_tokens`` less the pinned
    :data:`~paramem.utils.tokens.ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS``; a
    call whose payload tokens are zero or below takes no part (there is no
    payload to ratio against). Among the rest, this call's
    ``output_tokens`` and payload tokens are added into
    :attr:`Result.scan_reply_run_output_tokens` /
    :attr:`Result.scan_reply_run_payload_tokens` — the run totals whose
    ratio is the source for re-pinning
    :data:`~paramem.utils.tokens.ANONYMIZE_SCAN_REPLY_RATIO` — and,
    separately, this call's own ``output_tokens / payload_tokens`` updates
    :attr:`Result.largest_scan_reply_ratio` /
    :attr:`Result.largest_scan_reply_ratio_entry` when it exceeds the
    running maximum — the per-entry reading of the plateau
    (:data:`~paramem.utils.tokens.ANONYMIZE_SCAN_MAX_OUTPUT_TOKENS`), never
    itself the re-pin source.

    Also tallies how many scan calls this one entry cost: every
    ``contract.call_tokens`` record labelled ``"anonymize.scan"`` counts,
    regardless of its payload tokens (the local anonymizer now issues one
    scan call per turn of the payload, so this is the reader's count of
    those calls, not a ratio input). The count adds into
    :attr:`Result.scan_call_run_total` — the numerator for the run's mean
    calls-per-entry — and updates :attr:`Result.largest_scan_call_count` /
    :attr:`Result.largest_scan_call_count_entry` when it exceeds the
    running maximum.

    Runs :func:`_self_check` once the scorer's own occurrence walk is
    built — a disagreement, in either of its two arms, stops the run.
    """
    surfaces = entry_surfaces(entry)
    forward = contract.forward
    reverted_category = {
        d["text"]: d["category"] for d in contract.scan_dropped_entries if d["reason"] == "reverted"
    }
    unknown_values = _unknown_word_values(contract)

    scrubbed_occ = _occurrences_over_surfaces(surfaces, list(forward))
    reverted_occ = _occurrences_over_surfaces(surfaces, list(reverted_category))
    unknown_occ = _occurrences_over_surfaces(surfaces, unknown_values)

    _self_check(entry, forward, {o[1] for o in scrubbed_occ})

    for dropped in contract.scan_dropped_entries:
        if dropped["reason"] == "unknown_word":
            result.unknown_word_by_word[dropped["word"]] += 1

    scan_call_count = 0
    for call in contract.call_tokens:
        if call["label"] != "anonymize.scan":
            continue
        scan_call_count += 1
        payload_tokens = call["prompt_tokens"] - ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS
        if payload_tokens <= 0:
            continue
        result.scan_reply_run_output_tokens += call["output_tokens"]
        result.scan_reply_run_payload_tokens += payload_tokens
        ratio = call["output_tokens"] / payload_tokens
        if result.largest_scan_reply_ratio is None or ratio > result.largest_scan_reply_ratio:
            result.largest_scan_reply_ratio = ratio
            result.largest_scan_reply_ratio_entry = entry["id"]

    result.scan_call_run_total += scan_call_count
    if result.largest_scan_call_count is None or scan_call_count > result.largest_scan_call_count:
        result.largest_scan_call_count = scan_call_count
        result.largest_scan_call_count_entry = entry["id"]

    result.tot["entries"] += 1
    for gold in entry["gold"]:
        cat = gold["category"]
        turn_id, start, end = gold["turn"], gold["start"], gold["end"]
        same_turn_scrubbed = [o for o in scrubbed_occ if o[0] == turn_id]
        same_turn_unknown = [o for o in unknown_occ if o[0] == turn_id]
        caught = any(span_covers(s, e, start, end) for _, _, s, e in same_turn_scrubbed)
        partial = (not caught) and any(
            spans_overlap(s, e, start, end) for _, _, s, e in same_turn_scrubbed
        )

        if cat in configured:
            status = "caught" if caught else ("partial" if partial else "missed")
            result.tot["gold_in_scope"] += 1
            result.tot[f"{status}_in_scope"] += 1
            if cat == "Person":
                result.tot["gold_names"] += 1
                result.tot[f"{status}_names"] += 1
                result.by_lang[entry["lang"]]["gold_names"] += 1
                result.by_lang[entry["lang"]][f"{status}_names"] += 1
                if entry["casing"] == "lower":
                    result.by_casing["lower"]["gold_names"] += 1
                    result.by_casing["lower"][f"{status}_names"] += 1
            if cat in _CONTACT_PREFIXES:
                result.tot["gold_contact"] += 1
                result.tot[f"{status}_contact"] += 1
            if status == "missed":
                result.miss_list.append((entry["id"], turn_id, cat, gold["value"]))
            elif status == "partial":
                result.partial_list.append((entry["id"], turn_id, cat, gold["value"]))
        else:
            result.tot["gold_out_scope"] += 1
            same_turn_reverted = [o for o in reverted_occ if o[0] == turn_id]
            if caught or partial:
                result.tot["out_scope_scrubbed"] += 1
            elif any(spans_overlap(s, e, start, end) for _, _, s, e in same_turn_reverted):
                result.tot["out_scope_reversed"] += 1
            else:
                result.tot["out_scope_untagged"] += 1

        if not caught and any(spans_overlap(s, e, start, end) for _, _, s, e in same_turn_unknown):
            result.unknown_word_census[cat] += 1

    for value, placeholder in forward.items():
        v_occ = [o for o in scrubbed_occ if o[1] == value]
        overlaps_in_scope = any(
            o[0] == gold["turn"] and spans_overlap(o[2], o[3], gold["start"], gold["end"])
            for o in v_occ
            for gold in entry["gold"]
            if gold["category"] in configured
        )
        if overlaps_in_scope:
            result.tot["scrubbed_correct"] += 1
            continue
        wrong_cat = next(
            (
                gold["category"]
                for o in v_occ
                for gold in entry["gold"]
                if gold["category"] not in configured
                and o[0] == gold["turn"]
                and spans_overlap(o[2], o[3], gold["start"], gold["end"])
            ),
            None,
        )
        if wrong_cat is not None:
            result.tot["scrubbed_wrong_type"] += 1
            result.wrong_type_list.append((entry["id"], value, placeholder, wrong_cat))
        else:
            result.tot["scrubbed_junk"] += 1
            result.junk_list.append((entry["id"], value, placeholder))

    for decoy in entry["decoys"]:
        if decoy in forward:
            result.decoys_scrubbed.append((entry["id"], decoy))

    for turn_id, text in surfaces:
        if any(o[0] == turn_id and o[2] == 0 and o[3] == len(text) for o in scrubbed_occ):
            result.tot["whole_utterance_scrubs"] += 1

    result.tot["unsubstitutable"] += contract.inert_dropped
    result.unsub_list.extend(
        (entry["id"], d["text"]) for d in contract.scan_dropped_entries if d["reason"] == "inert"
    )


def score_corpus(
    entries: list[dict], contracts: dict[str, AnonymizedContract], configured: set[str]
) -> Result:
    """Score every entry with a completed (``status="ok"``) contract into
    one :class:`Result`.

    An entry whose contract is missing or did not complete
    (``opted_out``/``failed``) contributes no gold coverage and is listed
    in :attr:`Result.skipped` rather than silently dropped; its gold values
    are tallied into :attr:`Result.skipped_gold_names` /
    :attr:`Result.skipped_gold_contact` so the printed detail can state how
    much gold sits outside the recall percentages. The tally counts only
    gold categories in *configured* — the same gate the percentages' own
    denominators (``gold_names``/``gold_contact``) apply — so an
    out-of-scope category never inflates the excluded-gold count.
    """
    result = Result()
    for entry in entries:
        contract = contracts.get(entry["id"])
        if contract is None or contract.status != "ok":
            status = contract.status if contract is not None else "missing"
            result.skipped.append((entry["id"], status))
            for gold in entry["gold"]:
                if gold["category"] not in configured:
                    continue
                if gold["category"] == "Person":
                    result.skipped_gold_names += 1
                if gold["category"] in _CONTACT_PREFIXES:
                    result.skipped_gold_contact += 1
            continue
        score_entry(entry, contract, configured, result)
    return result


def pct(a: int, b: int) -> float | None:
    """*a* over *b* as a 0-100 float, or ``None`` when *b* is zero."""
    return 100.0 * a / b if b else None


def _fmt_pct(v: float | None) -> str:
    return f"{v:5.1f}%" if v is not None else " n/a "


_HIGHER_IS_BETTER = (
    "names_all",
    "names_lowercase",
    "names_en",
    "names_de",
    "names_fr",
    "names_es",
    "contact",
    "precision",
)
_LOWER_IS_BETTER = ("junk", "wrong_type", "partial", "unsubstitutable", "failed", "invented")


def scorecard_dict(r: Result) -> dict:
    """The scorecard's primary columns, as a plain dict — the shape both
    ``baseline.json`` and the regression check read.

    ``failed`` is the count of entries whose contract did not complete
    (every entry :func:`score_corpus` lists in :attr:`Result.skipped`) —
    lower is better, carried in the baseline file and the regression check
    alongside the other primary columns.

    ``invented`` is the count of ``reason="unknown_word"`` drop records
    over the run — the sum of :attr:`Result.unknown_word_by_word`'s values
    — lower is better, carried in the baseline file and the regression
    check alongside the other primary columns: a keyword outside the
    table is the model leaving the closed list.
    """
    tot = r.tot
    low = r.by_casing["lower"]
    scr = tot["scrubbed_correct"] + tot["scrubbed_wrong_type"] + tot["scrubbed_junk"]
    return {
        "names_all": pct(tot["caught_names"], tot["gold_names"]),
        "names_lowercase": pct(low["caught_names"], low["gold_names"]),
        "names_en": pct(r.by_lang["en"]["caught_names"], r.by_lang["en"]["gold_names"]),
        "names_de": pct(r.by_lang["de"]["caught_names"], r.by_lang["de"]["gold_names"]),
        "names_fr": pct(r.by_lang["fr"]["caught_names"], r.by_lang["fr"]["gold_names"]),
        "names_es": pct(r.by_lang["es"]["caught_names"], r.by_lang["es"]["gold_names"]),
        "contact": pct(tot["caught_contact"], tot["gold_contact"]),
        "precision": pct(tot["scrubbed_correct"], scr),
        "junk": tot["scrubbed_junk"],
        "wrong_type": tot["scrubbed_wrong_type"],
        "partial": tot["partial_in_scope"],
        "unsubstitutable": tot["unsubstitutable"],
        "failed": len(r.skipped),
        "invented": sum(r.unknown_word_by_word.values()),
    }


def regression_columns(current: dict, baseline: dict) -> list[str]:
    """Every primary column where *current* is worse than *baseline* — no
    threshold: any regression at all is named. ``None`` values (an empty
    denominator, e.g. no lowercase gold names in a pilot run) never
    compare as a regression.
    """
    bad = []
    for col in _HIGHER_IS_BETTER:
        c, b = current.get(col), baseline.get(col)
        if c is not None and b is not None and c < b:
            bad.append(col)
    for col in _LOWER_IS_BETTER:
        c, b = current.get(col), baseline.get(col)
        if c is not None and b is not None and c > b:
            bad.append(col)
    return bad


def print_detail(r: Result) -> None:
    """Print the run's aggregates and the FULL junk, miss, partial, inert
    and unknown-word lists, for the owner's reading.

    States the recall denominators explicitly: after the skipped-entry
    list, one line gives how many gold values the skipped (failed) entries
    hold — split into names (``Person``) and contact — so a recall
    percentage is always read beside the share of the corpus it does not
    cover.

    Also prints one scan-reply-ratio line carrying two distinct readings:
    the run ratio (:attr:`Result.scan_reply_run_output_tokens` over
    :attr:`Result.scan_reply_run_payload_tokens`) beside the pinned
    :data:`~paramem.utils.tokens.ANONYMIZE_SCAN_REPLY_RATIO` constant —
    this is the source to re-pin that constant from — and the largest
    single entry's own ratio (:attr:`Result.largest_scan_reply_ratio` /
    :attr:`Result.largest_scan_reply_ratio_entry`), which reads the
    output-reserve plateau
    (:data:`~paramem.utils.tokens.ANONYMIZE_SCAN_MAX_OUTPUT_TOKENS`)
    instead — a readout only, no threshold or warning logic.

    Also prints one scan-call-count line: the mean number of scan calls
    per scored entry (:attr:`Result.scan_call_run_total` over the scored
    entry count) beside the largest single entry's own count
    (:attr:`Result.largest_scan_call_count` /
    :attr:`Result.largest_scan_call_count_entry`) — the local anonymizer
    now issues one scan call per turn of the payload, so this is how many
    calls an entry cost, a readout only.
    """
    tot = r.tot
    print("\n" + "=" * 72)
    print("ANONYMIZER GATE — detail")
    print("=" * 72)
    print(f"  entries scored: {tot['entries']}   skipped: {len(r.skipped)}")
    if r.skipped:
        for entry_id, status in r.skipped:
            print(f"    skipped {entry_id}: status={status}")
    print(
        f"  skipped-entry gold outside the percentages: "
        f"names {r.skipped_gold_names}, contact {r.skipped_gold_contact}"
    )
    print(
        f"  in-scope gold {tot['gold_in_scope']} (names {tot['gold_names']}, "
        f"contact {tot['gold_contact']})   out-of-scope gold {tot['gold_out_scope']}"
    )
    print(
        f"  RECALL  in-scope caught {tot['caught_in_scope']}/{tot['gold_in_scope']} "
        f"(partial {tot['partial_in_scope']}, missed {tot['missed_in_scope']})"
    )
    scr = tot["scrubbed_correct"] + tot["scrubbed_wrong_type"] + tot["scrubbed_junk"]
    print(
        f"  SCRUBBED {scr}: correct {tot['scrubbed_correct']} | "
        f"wrong type {tot['scrubbed_wrong_type']} | junk {tot['scrubbed_junk']}   "
        f"precision {_fmt_pct(pct(tot['scrubbed_correct'], scr))}   "
        f"unsubstitutable {tot['unsubstitutable']}"
    )
    print(
        f"  OUT-OF-SCOPE gold: reversed {tot['out_scope_reversed']} | "
        f"scrubbed anyway {tot['out_scope_scrubbed']} | untagged {tot['out_scope_untagged']}"
    )
    print(f"  whole-utterance scrubs: {tot['whole_utterance_scrubs']}")
    print(f"  decoys scrubbed: {len(r.decoys_scrubbed)}")
    for entry_id, decoy in r.decoys_scrubbed:
        print(f"    {entry_id}: {decoy!r}")
    print(f"  unknown-word census by gold category: {dict(r.unknown_word_census)}")
    print(f"  unknown-word census by the model's own word: {dict(r.unknown_word_by_word)}")
    if r.largest_scan_reply_ratio is None:
        print("  scan reply ratio: n/a (no scan call had payload tokens)")
    else:
        run_ratio = r.scan_reply_run_output_tokens / r.scan_reply_run_payload_tokens
        print(
            f"  scan reply ratio: run {run_ratio:.2f} "
            f"(pinned ANONYMIZE_SCAN_REPLY_RATIO={ANONYMIZE_SCAN_REPLY_RATIO}); "
            f"largest single entry {r.largest_scan_reply_ratio:.2f} "
            f"({r.largest_scan_reply_ratio_entry})"
        )
    if tot["entries"] == 0:
        print("  scan calls per entry: n/a (no entries scored)")
    else:
        mean_scan_calls = r.scan_call_run_total / tot["entries"]
        print(
            f"  scan calls per entry: mean {mean_scan_calls:.2f} "
            f"over {tot['entries']} entries; largest single entry "
            f"{r.largest_scan_call_count} ({r.largest_scan_call_count_entry})"
        )

    print(f"\n  MISS list ({len(r.miss_list)}):")
    for entry_id, turn_id, cat, value in r.miss_list:
        print(f"    {entry_id} (turn {turn_id}): {cat} {value!r}")
    print(f"  PARTIAL list ({len(r.partial_list)}):")
    for entry_id, turn_id, cat, value in r.partial_list:
        print(f"    {entry_id} (turn {turn_id}): {cat} {value!r}")
    print(f"  JUNK list ({len(r.junk_list)}):")
    for entry_id, value, ph in r.junk_list:
        print(f"    {entry_id}: {value!r} -> {ph}")
    print(f"  WRONG-TYPE list ({len(r.wrong_type_list)}):")
    for entry_id, value, ph, cat in r.wrong_type_list:
        print(f"    {entry_id}: {value!r} -> {ph} (gold: {cat})")
    print(f"  UNSUBSTITUTABLE list ({len(r.unsub_list)}):")
    for entry_id, value in r.unsub_list:
        print(f"    {entry_id}: {value!r}")


def print_scorecard(scorecard: dict) -> None:
    """Print the primary-column scorecard row."""
    print("\n" + "=" * 72)
    print("SCORECARD — primary columns")
    print("=" * 72)
    for col in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER):
        value = scorecard.get(col)
        shown = _fmt_pct(value) if col in _HIGHER_IS_BETTER else str(value)
        print(f"  {col:<18} {shown}")


# ---------------------------------------------------------------------------
# Baseline (data/ha/calibration/anonymizer_gate/baseline.json).
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict | None:
    """The last accepted scorecard, or ``None`` when no baseline exists yet.

    Args:
        path: Baseline file path. No import-time default — ``main``
            resolves the module constant ``_BASELINE_PATH`` once and
            threads it through explicitly, so a test that redirects the
            baseline redirects it the same way it redirects every other
            disk access this tool makes.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(scorecard: dict, *, source: str, path: Path) -> None:
    """Record *scorecard* as the new accepted baseline, tagged with *source*.

    Args:
        scorecard: The primary-column scorecard to record.
        source: Free-text provenance for the accepted run (e.g. its run
            directory name).
        path: Baseline file path. No import-time default — see
            :func:`load_baseline`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**scorecard, "source": source}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _score_and_report(
    entries: list[dict],
    contracts: dict[str, AnonymizedContract],
    configured: set[str],
    run_dir: Path,
    baseline_path: Path,
    *,
    accept: bool,
    write_scorecard: bool,
) -> int:
    """The one scoring-and-verdict sequence: score *contracts*, print the
    detail and scorecard, compare against the last accepted baseline, and
    optionally accept a new one — reached by both the guarded (model-
    bearing) run and the score-only path (:func:`_score_from_disk`) so
    there is exactly one place this sequence is written.

    Args:
        entries: The (possibly ``--limit``-sliced) corpus entries.
        contracts: Every entry's completed contract, keyed by entry id —
            freshly run or loaded from a prior run's artifacts.
        configured: The active scrub categories' prefixes.
        run_dir: The run directory the scorecard is written beside.
        baseline_path: The accepted-baseline file path, resolved once by
            ``main`` from the module constant ``_BASELINE_PATH`` and
            threaded through explicitly.
        accept: Whether to record this run's scorecard as the new baseline
            (``--accept``; already refused together with ``--limit`` by the
            caller).
        write_scorecard: Whether to write ``run_dir/scorecard.json``.
            ``False`` when ``--limit`` sliced the entries, so a pilot
            slice never overwrites a complete run's own full-corpus
            scorecard.

    Returns:
        ``0`` always.

    Raises:
        AssertionError: ``score_corpus`` (via ``score_entry``'s
            ``_self_check``) raises when the scorer's own reconstruction
            of what substitutes disagrees with production's own
            substitution walk over the reconstructed payload text — never
            a normal scoring outcome.
    """
    result = score_corpus(entries, contracts, configured)
    print_detail(result)
    scorecard = scorecard_dict(result)
    print_scorecard(scorecard)

    baseline = load_baseline(baseline_path)
    if baseline is None:
        print("\nno accepted baseline on disk yet")
    else:
        regressed = regression_columns(scorecard, baseline)
        if regressed:
            print(f"\nREGRESSION on columns: {regressed}")
        else:
            print("\nno regression against the accepted baseline")

    if accept:
        write_baseline(scorecard, source=f"run {run_dir.name}", path=baseline_path)
        print(f"baseline accepted from run {run_dir.name}")

    if write_scorecard:
        (run_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Run-directory bookkeeping (per-entry artifacts, resumable).
# ---------------------------------------------------------------------------


def _new_run_dir(root: Path) -> Path:
    """A fresh, timestamped run directory path under *root*.

    Args:
        root: The run root. No import-time default — ``main`` resolves
            the module constant ``_RUN_ROOT`` once and threads it through
            explicitly, so a test that redirects the run root redirects
            it the same way it redirects every other disk access this
            tool makes.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def _latest_run_dir(root: Path) -> Path | None:
    """The most recent run directory under *root*, or ``None`` when *root*
    does not exist or holds none.

    Args:
        root: The run root. No import-time default — see :func:`_new_run_dir`.
    """
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def _entry_artifact_path(base_dir: Path, entry_id: str) -> Path:
    return base_dir / f"{entry_id}.json"


def _write_entry_artifact(
    base_dir: Path, entry_id: str, contract: AnonymizedContract, seconds: float
) -> None:
    """Write one entry's raw contract to disk, as the run proceeds — a
    crash loses only the entry in flight, never the entries already
    scored.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    _entry_artifact_path(base_dir, entry_id).write_text(
        json.dumps(
            {
                "id": entry_id,
                "status": contract.status,
                "failure": contract.failure,
                "forward": contract.forward,
                "scan_dropped_entries": contract.scan_dropped_entries,
                "inert_dropped": contract.inert_dropped,
                "model_calls": contract.model_calls,
                "call_tokens": list(contract.call_tokens),
                "raw": contract.raw,
                "seconds": seconds,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_entry_artifact(base_dir: Path, entry_id: str) -> AnonymizedContract | None:
    """Reconstruct the :class:`AnonymizedContract` fields the scorer reads
    from a previously written artifact — the ``--resume`` read path, and
    the score-only path's (:func:`_score_from_disk`) only source of
    contracts.
    """
    path = _entry_artifact_path(base_dir, entry_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnonymizedContract(
        status=data["status"],
        forward=data["forward"],
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        rekey_dropped=0,
        raw=data["raw"],
        failure=data["failure"],
        facts=[],
        model_calls=data["model_calls"],
        call_tokens=tuple(data["call_tokens"]),
        scan_dropped=0,
        scan_dropped_entries=data["scan_dropped_entries"],
        inert_dropped=data["inert_dropped"],
    )


def _run_dir_complete(entries: list[dict], run_dir: Path) -> bool:
    """Whether every entry in *entries* (after any ``--limit``) already has
    a written artifact under ``run_dir/entries`` — the score-only path's
    completeness test. A freshly created run directory (no ``--resume``,
    or ``--resume`` with nothing yet on disk) is never complete, so this
    always answers ``False`` for a run that has not actually finished. An
    empty *entries* list is never complete either — no entries were ever
    run, so there is nothing to score from disk.
    """
    entries_dir = run_dir / "entries"
    return bool(entries) and all(
        _entry_artifact_path(entries_dir, entry["id"]).exists() for entry in entries
    )


@dataclass
class RunRecord:
    """The provenance a run directory pins at creation: the scrub prefixes
    configured, the model id loaded, the ``--prompt-file`` path (or
    ``None`` for the shipped prompt home), and ``prompt_sha256`` — the
    sha256 hex digest of the prompt file's text, or ``None`` when no
    override was given. One shape serves both roles — what a fresh run
    directory writes (:func:`_write_run_record`) and what the current
    invocation's own values are compared against
    (:func:`_current_run_record`, :func:`_report_run_record`) — so there is
    no second JSON layout for the same provenance.
    """

    configured_prefixes: list[str]
    model_id: str
    prompt_file: str | None
    prompt_sha256: str | None = None


def _run_record_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def _current_run_record(
    *,
    configured: set[str],
    model_id: str,
    prompt_file: Path | None,
    prompt_override_text: str | None = None,
) -> RunRecord:
    """Build the :class:`RunRecord` for this invocation's own values —
    used both to write a fresh run directory's ``run.json`` and to compare
    against a prior run's recorded one. ``prompt_sha256`` is the sha256 hex
    digest of *prompt_override_text* when given, else ``None`` — the same
    digest a fresh run directory pins and a later score-from-disk
    invocation compares against.
    """
    return RunRecord(
        configured_prefixes=sorted(configured),
        model_id=model_id,
        prompt_file=str(prompt_file) if prompt_file is not None else None,
        prompt_sha256=(
            hashlib.sha256(prompt_override_text.encode("utf-8")).hexdigest()
            if prompt_override_text is not None
            else None
        ),
    )


def _write_run_record(run_dir: Path, record: RunRecord) -> None:
    """Write *record* to *run_dir*'s ``run.json`` — called once, by ``main``,
    only when it creates a fresh run directory. ``--resume`` onto an
    existing run directory never calls this, so the record always reflects
    the run's original configuration.
    """
    _run_record_path(run_dir).write_text(
        json.dumps(
            {
                "configured_prefixes": record.configured_prefixes,
                "model_id": record.model_id,
                "prompt_file": record.prompt_file,
                "prompt_sha256": record.prompt_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_run_record(run_dir: Path) -> RunRecord | None:
    """Read *run_dir*'s ``run.json``, or ``None`` when the directory holds
    no record file.
    """
    path = _run_record_path(run_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(
        configured_prefixes=data["configured_prefixes"],
        model_id=data["model_id"],
        prompt_file=data["prompt_file"],
        prompt_sha256=data["prompt_sha256"],
    )


def _report_run_record(run_dir: Path, current: RunRecord) -> None:
    """Print *run_dir*'s recorded provenance beside *current*'s own values,
    naming any field that differs — never a refusal; a complete run
    directory is always scored from its entry files as written, and the
    operator reads the mismatch to decide whether the re-score is
    meaningful. A run directory with no record file prints that fact
    instead and scores anyway.

    The prompt-file comparison is by content: ``prompt_sha256`` is the
    sha256 hex digest of the prompt override text (or ``None`` when no
    ``--prompt-file`` was given), so it names a difference whenever the
    override's actual content changed — including one side carrying no
    override at all — never merely whether a ``--prompt-file`` was given.
    """
    record = _read_run_record(run_dir)
    if record is None:
        print(f"run {run_dir.name} carries no run.json record")
        return

    print(
        f"run {run_dir.name} recorded: prefixes={record.configured_prefixes} "
        f"model_id={record.model_id!r} prompt_file={record.prompt_file!r}"
    )

    differing = []
    if current.configured_prefixes != record.configured_prefixes:
        differing.append("configured prefixes")
    if current.model_id != record.model_id:
        differing.append("model id")
    if current.prompt_sha256 != record.prompt_sha256:
        differing.append("prompt sha256")
    if differing:
        print(f"differs from this invocation: {', '.join(differing)}")


def _score_from_disk(
    entries: list[dict],
    model_cfg,
    configured: set[str],
    run_dir: Path,
    prompt_override_text: str | None,
    baseline_path: Path,
    prompt_file: Path | None,
    *,
    accept: bool,
    write_scorecard: bool,
) -> int:
    """Score a complete run directory from its entry artifacts alone.

    No GPU is acquired and no model is loaded: only the bare tokenizer
    (:func:`~paramem.models.loader.load_tokenizer`) for the skeleton
    re-measurement print every run makes. Every entry's contract comes from
    :func:`_load_entry_artifact`, over the same artifact reader
    ``--resume`` already uses — this is not a second reader. Delegates the
    scoring, detail print, scorecard, baseline comparison and ``--accept``
    handling to :func:`_score_and_report`, the one sequence the guarded
    (model-bearing) run also uses.

    Args:
        entries: The (possibly ``--limit``-sliced) corpus entries; every
            one of them is already known to have an artifact on disk (see
            :func:`_run_dir_complete`).
        model_cfg: The resolved model config, used only to load the bare
            tokenizer for the skeleton print — never a model.
        configured: The active scrub categories' prefixes.
        run_dir: The complete run directory being re-scored.
        prompt_override_text: The ``--prompt-file`` contents to measure the
            skeletons under, or ``None`` for the shipped prompt home.
        baseline_path: The accepted-baseline file path, resolved once by
            ``main`` and threaded through explicitly.
        prompt_file: This invocation's own ``--prompt-file`` path, or
            ``None`` — compared against the run directory's recorded
            provenance (see :func:`_report_run_record`).
        accept: Whether to record this run's scorecard as the new baseline.
        write_scorecard: Whether to write ``run_dir/scorecard.json``
            (``False`` under ``--limit``; see :func:`_score_and_report`).

    Returns:
        ``0`` always.

    Raises:
        AssertionError: Via :func:`_score_and_report`'s own scoring call —
            the scorer's self-check disagreeing with production's own
            substitution walk; never a normal scoring outcome.
    """
    from paramem.models.loader import load_tokenizer

    tokenizer = load_tokenizer(model_cfg)
    _print_skeleton_measurements(tokenizer, prompt_override_text=prompt_override_text)

    _report_run_record(
        run_dir,
        _current_run_record(
            configured=configured,
            model_id=model_cfg.model_id,
            prompt_file=prompt_file,
            prompt_override_text=prompt_override_text,
        ),
    )

    entries_dir = run_dir / "entries"
    contracts = {entry["id"]: _load_entry_artifact(entries_dir, entry["id"]) for entry in entries}
    print(
        f"run {run_dir.name} is complete on disk; scored from disk, "
        "no GPU acquired, no model loaded"
    )
    return _score_and_report(
        entries,
        contracts,
        configured,
        run_dir,
        baseline_path,
        accept=accept,
        write_scorecard=write_scorecard,
    )


# ---------------------------------------------------------------------------
# Entry point / dispatch — transcript path via anonymize_turn, facts path
# via anonymize() the way dispatch_anonymize_facts calls it.
# ---------------------------------------------------------------------------


def _run_transcript_entry(
    entry: dict,
    model,
    tokenizer,
    *,
    categories: tuple[ScrubCategory, ...],
    token_envelope: int,
) -> AnonymizedContract:
    """Run a transcript-shaped entry through ``anonymize_turn`` exactly as
    chat egress does: the entry's own ``history``, ``speaker_id`` and
    ``speaker_name``.
    """
    from paramem.graph.flows import anonymize_turn

    return anonymize_turn(
        entry["text"],
        model,
        tokenizer,
        history=entry.get("history", ()),
        speaker_id=entry.get("speaker_id"),
        speaker_name=entry.get("speaker_name"),
        categories=categories,
        token_envelope=token_envelope,
    )


def _run_facts_entry(
    entry: dict,
    model,
    tokenizer,
    *,
    categories: tuple[ScrubCategory, ...],
    token_envelope: int,
) -> AnonymizedContract:
    """Run a facts-shaped entry through ``anonymize()`` exactly the way
    ``paramem.server.calibrate.dispatch_anonymize_facts`` calls it:
    ``transcript=""``, ``identity_domain`` derived from the facts' own
    subject/object endpoints.
    """
    from paramem.cloud.anonymize import anonymize
    from paramem.graph.anonymizer_prompts import load_anonymizer_prompts

    facts = entry["facts"]
    identity_domain = sorted(
        {str(f.get("subject", "")) for f in facts if f.get("subject")}
        | {str(f.get("object", "")) for f in facts if f.get("object")}
    )
    anon_prompts = load_anonymizer_prompts()
    return anonymize(
        facts,
        model,
        tokenizer,
        transcript="",
        categories=categories,
        identity_domain=identity_domain,
        token_envelope=token_envelope,
        prompts=anon_prompts,
    )


def _run_entry(
    entry: dict, model, tokenizer, *, categories: tuple[ScrubCategory, ...], token_envelope: int
) -> AnonymizedContract:
    if "facts" in entry:
        return _run_facts_entry(
            entry, model, tokenizer, categories=categories, token_envelope=token_envelope
        )
    return _run_transcript_entry(
        entry, model, tokenizer, categories=categories, token_envelope=token_envelope
    )


def _run_corpus(
    entries: list[dict],
    model,
    tokenizer,
    *,
    categories: tuple[ScrubCategory, ...],
    token_envelope: int,
    run_dir: Path,
    resume: bool,
    cooldown_every: int,
) -> dict[str, AnonymizedContract]:
    """Run every entry, writing its raw contract to disk as it completes.

    ``resume`` skips an entry whose artifact is already on disk under
    *run_dir* ``/entries/``. ``cooldown_every`` (see :func:`_cooldown_if_due`)
    chunks the GPU burst: after every *cooldown_every* entries that
    actually issued a model call (a resumed entry never counts), the run
    pauses until the GPU has cooled.
    """
    contracts: dict[str, AnonymizedContract] = {}
    entries_dir = run_dir / "entries"
    model_calls_done = 0
    for entry in entries:
        if resume:
            existing = _load_entry_artifact(entries_dir, entry["id"])
            if existing is not None:
                contracts[entry["id"]] = existing
                print(f"  {entry['id']}: resumed from disk (status={existing.status})")
                continue
        start = time.perf_counter()
        contract = _run_entry(
            entry, model, tokenizer, categories=categories, token_envelope=token_envelope
        )
        seconds = time.perf_counter() - start
        _write_entry_artifact(entries_dir, entry["id"], contract, seconds)
        contracts[entry["id"]] = contract
        print(f"  {entry['id']}: status={contract.status} ({seconds:.2f}s)")
        model_calls_done += 1
        _cooldown_if_due(model_calls_done, every=cooldown_every)
    return contracts


def _run_payloads_mode(
    payloads_dir: Path,
    model,
    tokenizer,
    *,
    categories: tuple[ScrubCategory, ...],
    token_envelope: int,
    run_dir: Path,
    cooldown_every: int,
) -> None:
    """Run every ``graph_snapshot.json``-shaped file under *payloads_dir*
    through the fact path, local only, writing forward tables and drop
    records but scoring nothing (there is no gold for a real payload).
    ``cooldown_every`` chunks the GPU burst exactly as it does for
    :func:`_run_corpus` (see :func:`_cooldown_if_due`); a payload skipped
    before its model call (unreadable, no edges) never counts.
    """
    from paramem.server.calibrate import relations_from_snapshot

    out_dir = run_dir / "payloads"
    model_calls_done = 0
    for path in sorted(payloads_dir.glob("*.json")):
        try:
            facts = relations_from_snapshot(str(path))
        except Exception as exc:  # noqa: BLE001 — a malformed payload is reported, not fatal
            print(f"  {path.name}: could not read as a snapshot ({exc})")
            continue
        if not facts:
            print(f"  {path.name}: no edges, skipped")
            continue
        start = time.perf_counter()
        contract = _run_facts_entry(
            {"facts": facts}, model, tokenizer, categories=categories, token_envelope=token_envelope
        )
        seconds = time.perf_counter() - start
        _write_entry_artifact(out_dir, path.stem, contract, seconds)
        print(f"  {path.name}: status={contract.status} ({seconds:.2f}s, {len(facts)} facts)")
        model_calls_done += 1
        _cooldown_if_due(model_calls_done, every=cooldown_every)


def _wait_for_cooldown() -> None:
    """Block until the GPU has cooled to the project's everyday threshold.

    Goes through the repo's one cooldown gate
    (:func:`~paramem.training.thermal_throttle.wait_for_cooldown`), the same
    function ``scripts/dev/probe_orphan_classification_live.py`` calls, so
    there is no second cooldown implementation to keep in sync. That gate is
    bounded and returns the still-hot temperature on timeout; a GPU that has
    not reached the threshold by then is a thermal stall, and the run stops
    rather than continuing hot. The owner cools the chassis and continues the
    same run directory with ``--resume``.

    Raises:
        RuntimeError: The GPU stayed above the threshold for the whole bound.
    """
    temp = wait_for_cooldown(_COOLDOWN_THRESHOLD_C, _COOLDOWN_MAX_WAIT_S, label="anonymizer-gate")
    if temp is not None and temp > _COOLDOWN_THRESHOLD_C:
        raise RuntimeError(
            f"GPU cooldown stalled at {temp} C after {_COOLDOWN_MAX_WAIT_S} s "
            f"(threshold {_COOLDOWN_THRESHOLD_C} C); cool the chassis and continue with --resume"
        )


def _cooldown_if_due(count: int, *, every: int) -> None:
    """Pause for the GPU to cool every *every* model-bearing entries.

    Called after each entry that actually issued a model call — an entry
    resumed from disk (``--resume``) never advances *count*, since it
    never touches the model. Chunks a long GPU burst into ``every``-sized
    pieces so a full corpus run never accumulates thermal debt across
    hundreds of consecutive model calls.

    Args:
        count: The 1-based count of model-bearing entries processed so far
            in this run.
        every: The chunk size (``--cooldown-every``); a non-positive value
            never pauses (guarded by the CLI's own validation, but honored
            here too for direct callers).
    """
    if every > 0 and count % every == 0:
        print(f"  cooldown: {count} model-bearing entries done, waiting for the GPU to cool ...")
        _wait_for_cooldown()


def _prompt_override_context(prompt_override_text: str | None):
    """A ``prompt_overrides({"anonymization.txt": ...})`` context when
    *prompt_override_text* is given, else a no-op context — the one place
    a ``--prompt-file`` value becomes a prompt-home substitution, used by
    both :func:`_print_skeleton_measurements` and ``main``'s guarded
    (model-bearing) run.

    Args:
        prompt_override_text: The ``--prompt-file`` contents to substitute
            for ``anonymization.txt``, or ``None`` for the shipped prompt
            home.
    """
    from paramem.graph.prompts import prompt_overrides

    if prompt_override_text is None:
        return contextlib.nullcontext()
    return prompt_overrides({"anonymization.txt": prompt_override_text})


def _print_skeleton_measurements(tokenizer, *, prompt_override_text: str | None) -> None:
    """Re-measure the SCAN and ANCHOR prompt skeletons against their pinned
    reference constants and print one line per skeleton.

    Renders each section with an empty payload through the production
    section renderers (:func:`~paramem.cloud.anonymize_steps.render_scan_section`,
    :func:`~paramem.cloud.anonymize_steps.render_anchor_section`) and the
    production chat-wrapping render
    (:func:`~paramem.cloud.anonymize_steps.render_call_prompt`), counts
    each rendered prompt with *tokenizer* through
    :func:`~paramem.utils.tokens.encode_rendered` (never
    :func:`~paramem.utils.tokens.estimate_tokens`, whose word-count
    fallback would print as a measurement rather than an exact count), and
    prints the measured count, the pinned reference constant, and the
    signed difference for both skeletons — the operator decides whether to
    re-pin the constant, revert the edit, or accept the difference.

    Loads the anonymizer prompts inside the same
    :func:`_prompt_override_context` the corpus run itself uses when
    ``--prompt-file`` substitutes a variant, so a skeleton measured under
    an overridden prompt reflects the override, not the shipped file.

    Args:
        tokenizer: The tokenizer to measure with — the bare tokenizer from
            :func:`~paramem.models.loader.load_tokenizer` whenever no
            model is loaded (``--dry-run`` and the score-from-disk path
            alike), the loaded base model's tokenizer otherwise.
        prompt_override_text: The ``--prompt-file`` contents to substitute
            for ``anonymization.txt``, or ``None`` to measure the shipped
            prompt home.
    """
    from paramem.cloud.anonymize_steps import (
        render_anchor_section,
        render_call_prompt,
        render_scan_section,
    )
    from paramem.graph.anonymizer_prompts import load_anonymizer_prompts
    from paramem.utils.tokens import (
        ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS,
        ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS,
        encode_rendered,
    )

    with _prompt_override_context(prompt_override_text):
        prompts = load_anonymizer_prompts()
        scan_prompt = render_call_prompt(
            prompts.scan_system, render_scan_section(prompts.scan, ""), tokenizer
        )
        anchor_prompt = render_call_prompt(
            prompts.anchor_system,
            render_anchor_section(prompts.anchor, speaker_id="speaker1", values=(), text=""),
            tokenizer,
        )

    for label, rendered, reference in (
        ("SCAN", scan_prompt, ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS),
        ("ANCHOR", anchor_prompt, ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS),
    ):
        measured = len(encode_rendered(tokenizer, rendered)["input_ids"])
        diff = measured - reference
        print(f"{label} skeleton: measured={measured} reference={reference} diff={diff:+d}")


def _restore_server_gpu(
    *, server_base_url: str = _SERVER_BASE_URL, retries: int = 5, delay_seconds: float = 15.0
) -> None:
    """Ask the server to reacquire the GPU (``POST /gpu/acquire``), with
    bounded retry on a busy server: 503 while a fold runs, 409 during a
    base swap. The bearer comes from
    :func:`paramem.cli.http_client.resolve_token` — the CLI's own
    resolution order (ambient env, secret file, repo ``.env``).

    Args:
        server_base_url: The server's base URL — ``--server`` threaded
            through from :func:`main`, default the local server's address.
        retries: Maximum attempts before giving up.
        delay_seconds: Pause between retries on a busy server.

    Never raises: a run that leaves the server cloud-only is surfaced as a
    printed warning naming the door to call by hand, exactly as
    ``experiments.utils.gpu_guard.acquire_gpu`` already documents for the
    plain (non-restoring) guard.
    """
    from paramem.cli.http_client import (
        ServerHTTPError,
        ServerUnavailable,
        ServerUnreachable,
        post_json,
        resolve_token,
    )

    url = f"{server_base_url}/gpu/acquire"
    token = resolve_token()
    for attempt in range(retries):
        try:
            post_json(url, token=token, timeout=60.0)
            print("server GPU reacquired.")
            return
        except ServerHTTPError as exc:
            if exc.status_code in (503, 409) and attempt < retries - 1:
                print(
                    f"server busy ({exc.status_code} on {url}); "
                    f"retrying in {delay_seconds:.0f}s ..."
                )
                time.sleep(delay_seconds)
                continue
            print(
                f"WARNING: could not restore the server's GPU ({exc}); "
                f"POST {url} by hand once the server is free."
            )
            return
        except (ServerUnavailable, ServerUnreachable) as exc:
            print(
                f"WARNING: could not reach the server to restore its GPU ({exc}); "
                f"POST {url} by hand once the server is free."
            )
            return


def _positive_int(value: str) -> int:
    """``argparse`` ``type=`` validator: a positive integer, or
    ``argparse.ArgumentTypeError`` naming the rejected value.
    """
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return n


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser — every option this tool supports."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Registry entry name in paramem.server.config.MODEL_REGISTRY to load "
            "instead of the fixture's own model. Default: the model "
            "tests/fixtures/server.yaml configures (mistral)."
        ),
    )
    parser.add_argument(
        "--scrub",
        nargs="+",
        default=None,
        metavar="HINT",
        help=(
            "PII-vocabulary hints to scrub (e.g. --scrub 'person name' 'phone "
            "number'), resolved through resolve_scrub_categories -- probe a "
            "narrower or wider category set than the default. Default: the "
            "shipped default scrub set (SanitizationConfig()'s own dataclass "
            "default), never tests/fixtures/server.yaml's own sanitization.scrub, "
            "which is deliberately empty."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Path to an anonymization.txt-shaped prompt file to substitute for the "
            "shipped prompt home for the duration of this run, via "
            "paramem.graph.prompts.prompt_overrides()."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent run directory, skipping entries already on disk.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Run only the first N corpus entries (a pilot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Assemble every corpus entry and validate the corpus, then re-measure "
            "the anonymizer prompt skeletons with a CPU-only tokenizer load; exit "
            "without loading a model or touching the GPU."
        ),
    )
    parser.add_argument(
        "--payloads",
        type=Path,
        default=None,
        help=(
            "Directory of retained real fact payloads (graph_snapshot.json shape) "
            "to run through the fact path, local only, unscored."
        ),
    )
    parser.add_argument(
        "--accept",
        action="store_true",
        help=(
            "Record this run's scorecard as the new accepted baseline. Refused "
            "when --limit is set (a pilot's scorecard is not a corpus-wide "
            "baseline)."
        ),
    )
    parser.add_argument(
        "--server",
        default=_SERVER_BASE_URL,
        metavar="URL",
        help=(
            "Base URL of the running server, used to restore its GPU "
            f"(POST <URL>/gpu/acquire) after the run. Default: {_SERVER_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--cooldown-every",
        type=_positive_int,
        default=20,
        metavar="N",
        help=(
            "Pause for the GPU to cool (wait_for_cooldown) after every N "
            "model-bearing entries (default: 20). Ignored under --dry-run, "
            "which touches neither the GPU nor a model."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    ``--resume`` continues the most recent run directory. When every
    corpus entry (after any ``--limit``) already has a written artifact
    there (:func:`_run_dir_complete`), the run is scored straight from
    those artifacts (:func:`_score_from_disk`): no GPU is acquired and no
    model is loaded, only the bare tokenizer for the skeleton
    re-measurement print. Otherwise — a fresh run directory, or a
    ``--resume`` directory with entries still missing — the run proceeds
    under the GPU guard as normal, resuming only the entries not yet on
    disk. ``--accept`` records the resulting scorecard as the new baseline
    either way, and is refused together with ``--limit`` before either path
    is chosen.

    A freshly created run directory writes its own provenance
    (:class:`RunRecord` — configured prefixes, model id, prompt sha256) to
    ``run.json``; ``--resume`` onto an existing directory
    never rewrites it, and the score-from-disk path reports it beside this
    invocation's own values (see :func:`_report_run_record`). ``--limit``
    slices the entries scored but never writes ``run_dir/scorecard.json``
    — that file always reflects a full-corpus run.
    """
    args = build_arg_parser().parse_args(argv)

    if args.accept and args.limit is not None:
        print(
            "--accept is refused when --limit is set: a pilot is not a corpus-wide baseline.",
            file=sys.stderr,
        )
        return 1

    entries = load_corpus()
    try:
        validate_corpus(entries)
    except ValueError as exc:
        print(f"corpus invalid: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        entries = entries[: args.limit]

    try:
        categories = resolve_categories(args.scrub)
    except ValueError as exc:
        print(f"--scrub invalid: {exc}", file=sys.stderr)
        return 1
    configured = {c.prefix for c in categories}
    category_source = "--scrub override" if args.scrub is not None else "shipped default scrub set"

    from paramem.server.config import MODEL_REGISTRY, load_server_config

    server_cfg = load_server_config("tests/fixtures/server.yaml")
    if args.model is not None:
        if args.model not in MODEL_REGISTRY:
            print(
                f"unknown --model {args.model!r}; available: {sorted(MODEL_REGISTRY)}",
                file=sys.stderr,
            )
            return 1
        server_cfg.model_name = args.model
    model_cfg = server_cfg.model_config

    prompt_override_text = (
        args.prompt_file.read_text(encoding="utf-8") if args.prompt_file is not None else None
    )

    if args.dry_run:
        for entry in entries:
            entry_surfaces(entry)  # touches every entry's payload shape
        print(f"corpus valid: {len(entries)} entries assembled")
        print(f"configured categories ({category_source}): {sorted(configured) or '[]'}")

        from paramem.models.loader import load_tokenizer

        tokenizer = load_tokenizer(model_cfg)
        _print_skeleton_measurements(tokenizer, prompt_override_text=prompt_override_text)

        print("dry run complete; tokenizer loaded, no model, no GPU touched")
        return 0

    print(f"configured categories ({category_source}): {sorted(configured) or '[]'}")

    run_root = _RUN_ROOT
    baseline_path = _BASELINE_PATH
    write_scorecard = args.limit is None

    existing_run_dir = _latest_run_dir(run_root) if args.resume else None
    run_dir = existing_run_dir or _new_run_dir(run_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {run_dir}")

    if existing_run_dir is None:
        _write_run_record(
            run_dir,
            _current_run_record(
                configured=configured,
                model_id=model_cfg.model_id,
                prompt_file=args.prompt_file,
                prompt_override_text=prompt_override_text,
            ),
        )

    if args.payloads is None and _run_dir_complete(entries, run_dir):
        return _score_from_disk(
            entries,
            model_cfg,
            configured,
            run_dir,
            prompt_override_text,
            baseline_path,
            args.prompt_file,
            accept=args.accept,
            write_scorecard=write_scorecard,
        )

    import os

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    token_envelope = server_cfg.consolidation.extraction_anonymize_token_envelope
    print(f"model: {model_cfg.model_id}")

    from gpu_guard import GPUConfigMissing

    from experiments.utils.gpu_guard import acquire_gpu
    from paramem.models.loader import load_base_model

    contracts: dict[str, AnonymizedContract] | None = None
    try:
        with acquire_gpu(name="anonymizer-gate", interactive=False):
            _wait_for_cooldown()
            print("loading model ...")
            model, tokenizer = load_base_model(model_cfg, server_cfg.tier_config_map())
            print("model ready")
            _print_skeleton_measurements(tokenizer, prompt_override_text=prompt_override_text)

            with _prompt_override_context(prompt_override_text):
                if args.payloads is not None:
                    _run_payloads_mode(
                        args.payloads,
                        model,
                        tokenizer,
                        categories=categories,
                        token_envelope=token_envelope,
                        run_dir=run_dir,
                        cooldown_every=args.cooldown_every,
                    )
                else:
                    contracts = _run_corpus(
                        entries,
                        model,
                        tokenizer,
                        categories=categories,
                        token_envelope=token_envelope,
                        run_dir=run_dir,
                        resume=args.resume,
                        cooldown_every=args.cooldown_every,
                    )
    except GPUConfigMissing as exc:
        print(
            f"gpu-guard is not configured for this consumer: {exc}. Add a "
            "[consumers.anonymizer-gate] section to ~/.config/gpu-guard/config.toml.",
            file=sys.stderr,
        )
        return 1
    finally:
        print("restoring the server's GPU ...")
        _restore_server_gpu(server_base_url=args.server)

    if args.payloads is not None:
        print("payloads run complete (unscored).")
        return 0
    if contracts is None:
        return 1

    return _score_and_report(
        entries,
        contracts,
        configured,
        run_dir,
        baseline_path,
        accept=args.accept,
        write_scorecard=write_scorecard,
    )


if __name__ == "__main__":
    sys.exit(main())
