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

The tool always runs inside the experiment GPU guard (the server released
for the whole run) and restores the server's GPU with ``POST <--server
URL>/gpu/acquire`` in a ``finally``, retrying on a busy server (503 while a
fold runs, 409 during a base swap). ``--dry-run`` is the one mode that
touches neither the GPU nor a model — it only assembles the corpus and
validates its shape.

The corpus run chunks its GPU burst: every ``--cooldown-every`` (default
20) model-bearing entries, the tool pauses for the GPU to cool
(``wait_for_cooldown``) before continuing, the same chunked-burst pattern
any long GPU run in this project uses. ``--dry-run`` never runs a
model-bearing entry, so it never pauses.

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
# address together (paramem's Phone/Email/Address prefix rows) — never
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
        Active rows in schema row order.
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

    Runs :func:`_self_check` once the scorer's own occurrence walk is
    built — a disagreement, in either of its two arms, stops the run.
    """
    surfaces = entry_surfaces(entry)
    forward = contract.forward
    reverted_category = {
        d["text"]: d["category"] for d in contract.scan_dropped_entries if d["reason"] == "reverted"
    }
    unknown_values = _unknown_word_values(contract)

    candidate_values = list(dict.fromkeys([*forward.keys(), *reverted_category.keys()]))
    all_occ: list[tuple[object, str, int, int]] = []
    unknown_occ: list[tuple[object, str, int, int]] = []
    for turn_id, text in surfaces:
        for value, start, end in find_occurrences(text, candidate_values):
            all_occ.append((turn_id, value, start, end))
        for value, start, end in find_occurrences(text, unknown_values):
            unknown_occ.append((turn_id, value, start, end))

    scrubbed_occ = [o for o in all_occ if o[1] in forward]
    reverted_occ = [o for o in all_occ if o[1] in reverted_category]

    _self_check(entry, forward, {o[1] for o in scrubbed_occ})

    for dropped in contract.scan_dropped_entries:
        if dropped["reason"] == "unknown_word":
            result.unknown_word_by_word[dropped["word"]] += 1

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
    in :attr:`Result.skipped` rather than silently dropped.
    """
    result = Result()
    for entry in entries:
        contract = contracts.get(entry["id"])
        if contract is None or contract.status != "ok":
            status = contract.status if contract is not None else "missing"
            result.skipped.append((entry["id"], status))
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
_LOWER_IS_BETTER = ("junk", "wrong_type", "partial", "unsubstitutable")


def scorecard_dict(r: Result) -> dict:
    """The scorecard's primary columns, as a plain dict — the shape both
    ``baseline.json`` and the regression check read.
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


def load_baseline(path: Path = _BASELINE_PATH) -> dict | None:
    """The last accepted scorecard, or ``None`` when no baseline exists yet."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(scorecard: dict, *, source: str, path: Path = _BASELINE_PATH) -> None:
    """Record *scorecard* as the new accepted baseline, tagged with *source*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**scorecard, "source": source}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Run-directory bookkeeping (per-entry artifacts, resumable).
# ---------------------------------------------------------------------------


def _new_run_dir(root: Path = _RUN_ROOT) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def _latest_run_dir(root: Path = _RUN_ROOT) -> Path | None:
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
    from a previously written artifact — the ``--resume`` read path.
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
        type=int,
        default=None,
        help="Run only the first N corpus entries (a pilot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Assemble every corpus entry and validate the corpus; exit without "
            "loading a model or touching the GPU."
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
    """CLI entry point."""
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

    if args.dry_run:
        for entry in entries:
            entry_surfaces(entry)  # touches every entry's payload shape
        print(f"corpus valid: {len(entries)} entries assembled")
        print(f"configured categories ({category_source}): {sorted(configured) or '[]'}")
        print("dry run complete; no model loaded, no GPU touched")
        return 0

    import os

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

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
    token_envelope = server_cfg.consolidation.extraction_anonymize_token_envelope
    print(f"model: {model_cfg.model_id}")
    print(f"configured categories ({category_source}): {sorted(configured) or '[]'}")

    run_dir = (_latest_run_dir() if args.resume else None) or _new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {run_dir}")

    prompt_override_text = (
        args.prompt_file.read_text(encoding="utf-8") if args.prompt_file is not None else None
    )

    from gpu_guard import GPUConfigMissing

    from experiments.utils.gpu_guard import acquire_gpu
    from paramem.graph.prompts import prompt_overrides
    from paramem.models.loader import load_base_model

    contracts: dict[str, AnonymizedContract] | None = None
    try:
        with acquire_gpu(name="anonymizer-gate", interactive=False):
            _wait_for_cooldown()
            print("loading model ...")
            model, tokenizer = load_base_model(model_cfg, server_cfg.tier_config_map())
            print("model ready")

            override_ctx = (
                prompt_overrides({"anonymization.txt": prompt_override_text})
                if prompt_override_text is not None
                else contextlib.nullcontext()
            )
            with override_ctx:
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

    result = score_corpus(entries, contracts, configured)
    print_detail(result)
    scorecard = scorecard_dict(result)
    print_scorecard(scorecard)

    baseline = load_baseline()
    if baseline is None:
        print("\nno accepted baseline on disk yet")
    else:
        regressed = regression_columns(scorecard, baseline)
        if regressed:
            print(f"\nREGRESSION on columns: {regressed}")
        else:
            print("\nno regression against the accepted baseline")

    if args.accept:
        write_baseline(scorecard, source=f"run {run_dir.name}")
        print(f"baseline accepted from run {run_dir.name}")

    (run_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
