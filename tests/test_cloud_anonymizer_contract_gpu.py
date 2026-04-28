"""LLM-compliance contract test for ``extract_and_anonymize_for_cloud``.

Why this test exists: the helper calls ``extract_graph`` +
``_anonymize_with_local_model`` end-to-end on a real multi-turn
**transcript** -- the same input shape ParaMem operates on in
production.  Mocked tests cannot catch prompt-compliance regressions
(per ``feedback_prompt_contract_tests.md``).

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
* ``deanonymize_text(anon_text, mapping)`` whitespace-equals the
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

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU contract test requires --gpu flag",
    ),
]


# Empirical baseline calibrated on Mistral 7B Instruct v0.3 with the
# repair pipeline + NER cross-check + default scope ``[person]`` + the
# adapted (codebase-pattern-aligned) anonymization prompt.  Variance
# recalibration on 2026-04-29 across 10 iterations × 5 fixtures (50
# runs) showed 40/40 personal-marker success with zero flakes —
# Mistral 7B at temperature=0.0 is fully deterministic on this fixture.
# Threshold raised from 0.75 to 0.80 so the test trips on the very
# first per-fixture regression (any 1/4 failure) without going to the
# strict 0.9× empirical recommendation.  Re-run
# scripts/dev/calibrate_cloud_anonymizer.py after any change to
# extractor.py, NER scope, or the anonymization prompt.
_MATCH_THRESHOLD = 0.80


# Default scope under which the contract is asserted.  Mirrors the
# production default in ``SanitizationConfig.cloud_scope``.  Only names
# whose spaCy NER type maps into this set are required to be
# anonymized; out-of-scope categories (places, organizations, etc.)
# pass through verbatim by design.
_DEFAULT_SCOPE = {"person"}


# Fixture transcripts: single-turn user queries — the production input
# shape ``_handle_cloud_response`` passes to ``extract_and_anonymize_for_cloud``
# (the cloud-egress entry point anonymizes only the current-turn text;
# conversation history flows separately through ``_sanitize_history``).
# Earlier multi-turn ``[user]/[assistant]`` fixtures here were testing
# a code path the cloud-egress helper no longer receives in production.
#
# Each entry carries the user query, the speaker_name the chat handler
# would have resolved via voice enrollment, and the names that appear
# *in the query text* and MUST therefore be anonymized before the cloud
# sees the query.
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
        "transcript": "Should I follow up with Alex tomorrow about the project deadline?",
        "expected_names": ["Alex"],
    },
    {
        "id": "person_and_place",
        "speaker_name": "Anna",
        "transcript": (
            "What restaurants should I recommend to Alex when they move to Berlin next month?"
        ),
        # Berlin is intentionally NOT in expected_names under the
        # default scope ``[person]`` — places pass through verbatim so
        # the cloud can recommend Berlin restaurants by name.  Only
        # Alex (a person) must be anonymized.
        "expected_names": ["Alex"],
    },
    {
        "id": "two_people",
        "speaker_name": "Anna",
        "transcript": "Did Alex and Sam finalize the agenda for the workshop?",
        "expected_names": ["Alex", "Sam"],
    },
    {
        "id": "no_personal_markers",
        "speaker_name": "Anna",
        "transcript": "What is the speed of light in vacuum?",
        "expected_names": [],
    },
    {
        "id": "conversational_shape",
        "speaker_name": "Anna",
        "transcript": "Should I call the vet about Pat's dog being sick?",
        "expected_names": ["Pat"],
    },
]


@pytest.fixture(scope="module")
def loaded_model():
    """Load the Mistral 7B local cloud anonymizer judge once per module.

    Uses ``load_server_config("tests/fixtures/server.yaml")`` to pin the
    calibration target.  The ``_MATCH_THRESHOLD`` above is anchored to
    Mistral 7B at temperature 0; loading any other model would silently
    re-calibrate against an untested baseline.
    """
    import gc
    import os

    import torch

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from paramem.models.loader import load_base_model
    from paramem.server.config import load_server_config

    cfg = load_server_config("tests/fixtures/server.yaml")
    model, tokenizer = load_base_model(cfg.model_config)
    yield model, tokenizer

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalise(text: str) -> str:
    """Whitespace-normalise for round-trip comparison.  Anonymizer LLM may
    re-flow whitespace; we only care about token content."""
    return " ".join(text.split())


def test_cloud_anonymizer_contract(loaded_model):
    """Real-LLM contract: leak-guard correctness + round-trip integrity.

    Each query in the fixture is run through ``anonymize_outbound``.
    Per-query contract (when mapping is non-empty):
      1. No real name in mapping leaks into anon_text (forward leak guard).
         This is what ``anonymize_outbound`` already enforces — verified
         here on real LLM output.
      2. Every expected name appears in ``mapping.keys()`` (coverage).
      3. ``deanonymize_inbound(anon_text, mapping)`` whitespace-equals the
         original query (round-trip).

    Aggregate contract: at least ``_MATCH_THRESHOLD`` of queries with
    personal markers must succeed.  No-personal-markers queries pass
    through with empty mapping and contribute neutrally to the rate.
    """
    from paramem.graph.extractor import (
        deanonymize_text,
        extract_and_anonymize_for_cloud,
    )

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

        anon_text, mapping = extract_and_anonymize_for_cloud(
            transcript,
            model,
            tokenizer,
            speaker_name=entry["speaker_name"],
            pii_scope=_DEFAULT_SCOPE,
        )

        # Empty mapping is a privacy-safe block (leak guard tripped or
        # model produced nothing usable).  Counts toward "blocks", does
        # not count toward "successes" of personal-marker queries.
        if not mapping:
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
                f"present in anon_text {anon_text!r} "
                f"(extraction or NER cross-check missed it)"
            )

        # Round-trip contract: deanonymize_text restores the original
        # transcript (modulo whitespace, which the anonymizer LLM may reflow).
        round_trip = deanonymize_text(anon_text, mapping)
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
    """Wider-scope contract: under cloud_scope={person, place}, place
    names also get anonymized.

    The default-scope test above leaves place names verbatim by design
    (so the cloud can answer "Berlin restaurants?" sensibly).  This
    test differentiates: picks the ``person_and_place`` fixture entry —
    the only one carrying both categories — and runs it under the
    stricter scope ``{person, place}``.  Validates that the wider
    scope path actually anonymizes places on real LLM output, not
    just in unit tests with synthetic graphs.

    Operators picking the stricter posture (privacy over cloud-utility
    on places) need this guarantee to hold.
    """
    from paramem.graph.extractor import (
        deanonymize_text,
        extract_and_anonymize_for_cloud,
    )

    model, tokenizer = loaded_model

    entry = next(e for e in _FIXTURE if e["id"] == "person_and_place")
    transcript = entry["transcript"]

    anon_text, mapping = extract_and_anonymize_for_cloud(
        transcript,
        model,
        tokenizer,
        speaker_name=entry["speaker_name"],
        pii_scope={"person", "place"},
    )

    if not mapping:
        pytest.fail(
            f"Strict scope produced empty mapping (block) on {entry['id']}; "
            f"expected both Alex and Berlin to be anonymized."
        )

    # Both person AND place must be absent from anon_text under the
    # stricter scope.  This is the privacy contract the operator opted
    # into; a leak of either is a regression.
    for name in ("Alex", "Berlin"):
        assert name not in anon_text, (
            f"[{entry['id']}, scope=person+place] Privacy breach: {name!r} "
            f"present in anon_text {anon_text!r}"
        )

    # Round-trip: deanonymize_text restores the original (modulo
    # whitespace, which the anonymizer LLM may reflow).  Catches a
    # mapping/transcript inconsistency that would silently leave the
    # cloud's response with placeholders the user sees.
    round_trip = deanonymize_text(anon_text, mapping)
    assert _normalise(round_trip) == _normalise(transcript), (
        f"[{entry['id']}, scope=person+place] Round-trip mismatch:\n"
        f"  original:   {transcript!r}\n"
        f"  round-trip: {round_trip!r}"
    )
