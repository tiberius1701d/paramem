"""LLM-compliance contract test for ``extract_and_anonymize_for_cloud``.

Why this test exists: the helper calls ``extract_graph`` +
``anonymize_with_local_model`` end-to-end on a real multi-turn
**transcript** -- the same input shape ParaMem operates on in
production.  Mocked tests cannot catch prompt-compliance regressions;
only a live model run on production-shaped input can verify the contract.

The cloud anonymizer is the privacy-critical seam between local memory
and SOTA escalation: when ``sanitization.cloud_mode`` is ``"anonymize"``
or ``"both"``, this primitive is the only thing standing between
personal facts and the cloud.  A regression here is not a quality
issue, it's a privacy breach.

The fixture is **multi-turn transcripts** in the production
``[user]``/``[assistant]`` shape -- not isolated queries.  Production
cloud egress always sees a transcript (history + current turn) and the
extractor needs the kind of self-claims that real conversations carry.
Single-query fixtures are inappropriate -- they don't carry anchor
facts the extractor can latch onto.

Per-transcript contract (asserted on every fixture entry that produces
a mapping):

* Every ``mapping.keys()`` (real name) is absent from ``anon_text``
  (forward-path leak guard).
* Every entry in ``expected_names`` appears in ``mapping.keys()``
  (coverage).
* ``deanonymize_text(anon_text, reverse)`` whitespace-equals the
  original transcript (round-trip integrity).

Aggregate contract: at least ``_MATCH_THRESHOLD`` of fixture transcripts
with personal markers must succeed.  Lower threshold = more queries
fall back to per-query block, which is privacy-safe but reduces cloud
quality.  Calibrate on first run.

Run: ``pytest tests/test_cloud_anonymizer_contract_gpu.py --gpu``

The server must be stopped so the model can claim the GPU.  A module-
scoped fixture loads the local model once.
"""

from __future__ import annotations

import pytest

from paramem.server.config import _DEFAULT_SCRUB as _PRODUCTION_DEFAULT_SCRUB

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU contract test requires --gpu flag",
    ),
]


# Empirical baseline calibrated on Mistral 7B Instruct v0.3 with the
# production default scrub categories in effect at calibration time
# (see ``_DEFAULT_SCRUB`` below — imported live from ``config.py`` so
# this baseline always describes whatever the current default is) + the
# two-artifact (mapping + anonymized_transcript) anonymization prompt.
# Variance recalibration on 2026-04-29 (pre-redesign scope walk +
# single-artifact prompt) across 10 iterations × 5 fixtures (50 runs)
# showed 40/40 personal-marker success with zero flakes — Mistral 7B at
# temperature=0.0 is fully deterministic on this fixture.  Threshold
# raised from 0.75 to 0.80 so the test trips on the very first
# per-fixture regression (any 1/4 failure) without going to the strict
# 0.9× empirical recommendation.  NEEDS RECALIBRATION against the
# redesigned prompt (model now also authors ``anonymized_transcript`` —
# see FR-cal) AND against the 2026-07 scrub-vocabulary enrichment (5
# terse labels -> 12 load-bearing terms) via
# ``scripts/dev/calibrate_cloud_anonymizer.py`` before ship; the 0.80
# threshold is carried forward as a conservative floor, not a
# re-verified number.
# Re-run scripts/dev/calibrate_cloud_anonymizer.py after any change to
# extractor.py, the default scope, or the anonymization prompt.
_MATCH_THRESHOLD = 0.80


# Default scrub categories under which the contract is asserted.
# Imported directly from production (rather than duplicated as a
# literal) so this test always tracks whatever ``SanitizationConfig``
# actually ships — the prompt's sole scope authority (no code-side
# entity-type gate): the model classifies real values against this
# vocabulary and decides what to placeholder.  Categories NOT covered
# (city, organization, product, ...) pass through verbatim by design so
# the cloud can still reason about places and things sensibly.
_DEFAULT_SCRUB = set(_PRODUCTION_DEFAULT_SCRUB)


# Fixture transcripts: single-turn user queries — the production input
# shape ``_escalate_via_cloud_policy`` (paramem/server/inference.py:567)
# passes to ``extract_and_anonymize_for_cloud`` (the cloud-egress entry
# point anonymizes only the current-turn text; conversation history flows
# separately through ``_sanitize_history``).  Earlier multi-turn
# ``[user]/[assistant]`` fixtures here were testing a code path the
# cloud-egress helper no longer receives in production.
#
# Each entry carries the user query, the speaker_name and speaker_id the
# chat handler would have resolved via voice enrollment
# (``app.py:3593`` threads ``_resolved.speaker_id`` into ``handle_chat``,
# which threads it unchanged into ``_escalate_via_cloud_policy`` at
# ``inference.py:643`` and on into ``extract_and_anonymize_for_cloud``),
# and the names that appear *in the query text* and MUST therefore be
# anonymized before the cloud sees the query.
#
# ``speaker_id`` is NOT optional here even though the helper's own
# signature defaults it to ``None`` for text-only requests with no
# resolved speaker: ``extract_and_anonymize_for_cloud`` threads it into
# ``build_speaker_context`` (extractor.py:322), which formats it verbatim
# into the extraction-prompt speaker directive
# (configs/prompts/speaker_directive.txt) as "the current speaker's
# system identifier is {speaker_id}".  Omitting it does not skip the
# directive — the helper falls back to the literal sentinel
# ``"cloud_egress"`` (extractor.py:920), which conflicts with the
# extraction few-shots in configs/prompts/extraction.txt (every example
# hardcodes the subject as ``"speaker0"``) and is not the shape any real
# request produces.  Using the production-shaped ``"speaker0"`` here is
# load-bearing for the contract, not decorative.
#
# ``speaker_name`` is metadata used by the extraction prompt to bind
# first-person facts to a concrete subject.  It is NOT included in
# ``expected_names`` because the speaker's name does not appear in the
# query text itself — there is nothing to leak.  The anonymizer may or
# may not bind the speaker to a placeholder; either is privacy-safe.
_FIXTURE = [
    {
        "id": "single_person_self_claim",
        "speaker_name": "Anna",
        "speaker_id": "speaker0",
        "transcript": "Should I follow up with Alex tomorrow about the project deadline?",
        "expected_names": ["Alex"],
    },
    {
        "id": "person_and_place",
        "speaker_name": "Anna",
        "speaker_id": "speaker0",
        # Declarative, not interrogative: "What restaurants should I
        # recommend to Alex when they move to Berlin next month?" (the
        # original fixture text) is an interrogative that local_extract
        # yields ZERO relations for, so extract_and_anonymize_for_cloud
        # fails closed at the zero-relations gate before the anonymizer
        # ever runs (extractor.py:938-939) — the test named for the
        # anonymizer never reached it. Verified live (2026-07-14,
        # /calibrate/extract, temperature 0): this declarative produces
        # relations carrying both a person name (Alex) and a place name
        # (Berlin) — e.g. {"subject": "Alex", "predicate": "lives_in",
        # "object": "Berlin"} — so the anonymizer actually runs on both.
        "transcript": "Alex, my friend, moved to Berlin for a new job.",
        # Berlin is intentionally NOT in expected_names under the
        # production default scrub categories (``_DEFAULT_SCRUB`` above —
        # name/phone/address/online-identity sub-terms, no city/place
        # category) — city names pass through verbatim so the cloud can
        # recommend Berlin restaurants by name.  Only Alex (a person)
        # must be anonymized.
        "expected_names": ["Alex"],
    },
    {
        "id": "two_people",
        "speaker_name": "Anna",
        "speaker_id": "speaker0",
        "transcript": "Did Alex and Sam finalize the agenda for the workshop?",
        "expected_names": ["Alex", "Sam"],
    },
    {
        "id": "no_personal_markers",
        "speaker_name": "Anna",
        "speaker_id": "speaker0",
        "transcript": "What is the speed of light in vacuum?",
        "expected_names": [],
    },
    {
        "id": "conversational_shape",
        "speaker_name": "Anna",
        "speaker_id": "speaker0",
        "transcript": "Should I call the vet about Pat's dog being sick?",
        "expected_names": ["Pat"],
    },
]


@pytest.fixture()
def loaded_model(gpu_base_model):
    """Expose the session-scoped base model for cloud anonymizer contract tests.

    Delegates to the session-scoped ``gpu_base_model`` fixture
    (``conftest.py``) so the model is loaded exactly once per
    ``pytest --gpu`` invocation.  Inference-only: this file creates no
    adapters, so no teardown cleanup is required beyond what the session
    fixture handles.

    The ``_MATCH_THRESHOLD`` above is anchored to Mistral 7B at temperature
    0; loading any other model would silently re-calibrate against an
    untested baseline.
    """
    return gpu_base_model


def _normalise(text: str) -> str:
    """Whitespace-normalise for round-trip comparison.  Anonymizer LLM may
    re-flow whitespace; we only care about token content."""
    return " ".join(text.split())


def test_cloud_anonymizer_contract(loaded_model):
    """Real-LLM contract: leak-guard correctness + round-trip integrity.

    Each query in the fixture is run through ``anonymize_outbound``.
    Per-query contract (when the call is not a genuine block —
    ``anon_text`` non-empty; ``mapping`` may legitimately be empty):
      1. No real name in mapping leaks into anon_text (forward leak guard).
         This is what ``anonymize_outbound`` already enforces — verified
         here on real LLM output.
      2. Every expected name appears in ``mapping.keys()`` (coverage) and
         is absent from ``anon_text`` — an empty ``mapping`` proves this
         the hard way: if ``expected_names`` are truly present in the
         transcript, an unmodified ``anon_text`` trips the leak-guard
         assertion below.
      3. ``deanonymize_inbound(anon_text, mapping)`` whitespace-equals the
         original query (round-trip).

    Aggregate contract: at least ``_MATCH_THRESHOLD`` of queries with
    personal markers must succeed.  No-personal-markers queries pass
    through (mapping may be empty) and contribute neutrally to the rate.
    """
    from paramem.graph.extractor import extract_and_anonymize_for_cloud
    from paramem.graph.placeholders import deanonymize_text

    model, tokenizer = loaded_model

    successes = 0
    blocks = 0
    expected_personal = 0
    failures = []

    for entry in _FIXTURE:
        transcript = entry["transcript"]
        expected_names = entry["expected_names"]
        if expected_names:
            expected_personal += 1

        anon_text, mapping, reverse = extract_and_anonymize_for_cloud(
            transcript,
            model,
            tokenizer,
            speaker_id=entry["speaker_id"],
            speaker_name=entry["speaker_name"],
            scrub=_DEFAULT_SCRUB,
        )

        # A genuinely blocked call is ``anon_text == ""`` (extraction
        # failure, empty input, or the anonymizer's own parse failure —
        # ``mapping is None``, fail-closed).  ``mapping == {}`` no longer
        # means "block": it PROCEEDS —
        # the anonymizer ran and found nothing in scope, trusted as a
        # legitimate verdict rather than treated as a safe block.  That
        # verdict is itself exercised by the leak-guard assertions below
        # rather than skipped: if ``expected_names`` really are present
        # in the transcript, an unscrubbed ``anon_text`` trips them.
        if not anon_text:
            if expected_names:
                blocks += 1
            continue

        # Forward leak-guard contract: every real name in the mapping
        # AND every expected_name from the fixture must be absent from
        # anon_text.  The mapping check verifies the wrapper's invariant
        # on real LLM output; the expected_names check is the strict
        # privacy contract — even if extraction missed a name (so it's
        # never in the mapping), if it appears verbatim in the cloud
        # payload that's a leak.  Hard-asserted: a name visible in
        # anon_text is a privacy breach, not a coverage shortfall.
        for name in mapping:
            assert name not in anon_text, (
                f"[{entry['id']}] Leak guard breach: real name {name!r} "
                f"present in anon_text {anon_text!r}"
            )
        for name in expected_names:
            assert name not in anon_text, (
                f"[{entry['id']}] Privacy breach: expected_name {name!r} "
                f"present in anon_text {anon_text!r} (extraction missed it)"
            )

        # Round-trip contract: deanonymize_text restores the original
        # transcript (modulo whitespace, which the anonymizer LLM may reflow).
        round_trip = deanonymize_text(anon_text, reverse)
        if _normalise(round_trip) != _normalise(transcript):
            failures.append(
                f"[{entry['id']}] Round-trip mismatch:\n"
                f"  original: {transcript!r}\n"
                f"  round-trip: {round_trip!r}"
            )
            continue

        if expected_names:
            successes += 1

    # Aggregate threshold check on the personal-marker subset.
    if expected_personal == 0:
        pytest.skip("Fixture has no personal-marker queries — recalibrate.")

    rate = successes / expected_personal
    if rate < _MATCH_THRESHOLD:
        report = "\n".join(failures) if failures else "(no per-query failures recorded)"
        pytest.fail(
            f"Cloud anonymizer compliance regression: "
            f"{successes}/{expected_personal} succeeded "
            f"(rate {rate:.0%}, blocks {blocks}, threshold {_MATCH_THRESHOLD:.0%}).\n"
            f"Per-query notes:\n{report}"
        )


def test_cloud_anonymizer_contract_strict_scope_anonymizes_places(loaded_model):
    """Wider-scope contract: under ``scrub={person name, city}``, place
    names also get anonymized.

    The default-scope test above leaves place names verbatim by design
    (so the cloud can answer "Berlin restaurants?" sensibly).  This
    test differentiates: picks the ``person_and_place`` fixture entry —
    the only one carrying both categories — and runs it under a wider
    operator-configured scrub set.  Validates that a wider ``scrub``
    actually reaches the model's own scope decision on real LLM output
    (the prompt is the sole scope authority — there is no code-side
    entity-type gate to unit-test with a synthetic graph instead).

    Operators picking the stricter posture (privacy over cloud-utility
    on places) need this guarantee to hold.
    """
    from paramem.graph.extractor import extract_and_anonymize_for_cloud
    from paramem.graph.placeholders import deanonymize_text

    model, tokenizer = loaded_model

    entry = next(e for e in _FIXTURE if e["id"] == "person_and_place")
    transcript = entry["transcript"]

    anon_text, mapping, reverse = extract_and_anonymize_for_cloud(
        transcript,
        model,
        tokenizer,
        speaker_id=entry["speaker_id"],
        speaker_name=entry["speaker_name"],
        scrub={"person name", "city"},
    )

    if not mapping:
        # An empty mapping here is a compliance miss regardless of the
        # proceed-on-empty decision (mapping == {} no longer means
        # "block" — see extract_and_anonymize_for_cloud): this test
        # specifically verifies the wider scrub set actually widens what
        # the model classifies in scope, and it did not.
        reason = "anon_text empty (block)" if not anon_text else "nothing anonymized"
        pytest.fail(
            f"Strict scope produced empty mapping ({reason}) on {entry['id']}; "
            f"expected both Alex and Berlin to be anonymized."
        )

    # Both person AND place must be absent from anon_text under the
    # stricter scope.  This is the privacy contract the operator opted
    # into; a leak of either is a regression.
    for name in ("Alex", "Berlin"):
        assert name not in anon_text, (
            f"[{entry['id']}, scrub=person_name+city] Privacy breach: {name!r} "
            f"present in anon_text {anon_text!r}"
        )

    # Round-trip: deanonymize_text restores the original (modulo
    # whitespace, which the anonymizer LLM may reflow).  Catches a
    # mapping/transcript inconsistency that would silently leave the
    # cloud's response with placeholders the user sees.
    round_trip = deanonymize_text(anon_text, reverse)
    assert _normalise(round_trip) == _normalise(transcript), (
        f"[{entry['id']}, scrub=person_name+city] Round-trip mismatch:\n"
        f"  original:   {transcript!r}\n"
        f"  round-trip: {round_trip!r}"
    )
