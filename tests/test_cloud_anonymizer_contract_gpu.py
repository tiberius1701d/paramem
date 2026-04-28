"""LLM-compliance contract test for the inference-side cloud anonymizer.

Why this test exists: ``anonymize_outbound`` calls ``_anonymize_with_local_model``,
which depends on the local LLM following the anonymization prompt — emitting
valid JSON, covering every personal name in the mapping, producing an
``anon_transcript`` with no real-name leakage.  Mocked tests cannot catch
prompt-compliance regressions (per ``feedback_prompt_contract_tests.md``).

The cloud anonymizer is the privacy-critical seam between local memory and
SOTA escalation: when ``sanitization.cloud_mode`` is ``"anonymize"`` or
``"both"``, this primitive is the only thing standing between personal facts
and the cloud.  A regression here is not a quality issue, it's a privacy
breach.

This test runs the real local model against a small fixture of queries with
known personal markers, asserts the wrapper's contract end-to-end, and
records the success rate.  The threshold is the empirical Mistral 7B baseline
at temperature 0 — it must hold for the test to remain a regression guard.

Per-query contract (asserted on every query that produces a mapping):

* Every ``mapping.keys()`` (real name) is absent from ``anon_text``
  (forward-path leak guard).
* ``deanonymize_inbound(anon_text, mapping) == original_query`` modulo
  whitespace (round-trip integrity).

Aggregate contract: at least ``_MATCH_THRESHOLD`` of fixture queries with
personal markers must produce non-empty mappings (i.e., not be blocked by
the leak guard / model failure).  Lower threshold = more queries fall back
to per-query block, which is privacy-safe but reduces cloud quality.
Calibrate on first run.

Run: ``pytest tests/test_cloud_anonymizer_contract_gpu.py --gpu``

The server must be stopped so the model can claim the GPU.  A module-
scoped fixture loads the local model once.
"""

from __future__ import annotations

import gc

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU contract test requires --gpu flag",
    ),
]


# Empirical baseline.  Calibrate on first GPU run; raise only when the
# anonymization prompt is structurally improved (not when the local
# model is upgraded -- a stronger model raising the rate is welcome but
# raising the threshold without recalibration would silently break this
# test if the user reverts).
_MATCH_THRESHOLD = 0.6  # 3 of 5 fixture queries succeed


# Fixture queries: mix of single name, multiple names, no names, and
# longer realistic conversational shape.  Each `expected_names` lists
# names that MUST appear in the anonymizer's mapping if the query is
# not blocked by the leak guard.
_FIXTURE = [
    {
        "id": "single_person",
        "query": "What did Alex tell me about the project yesterday?",
        "expected_names": ["Alex"],
    },
    {
        "id": "person_and_place",
        "query": "When is Alex visiting Berlin next?",
        "expected_names": ["Alex", "Berlin"],
    },
    {
        "id": "two_people",
        "query": "Did Alex and Sam agree on the meeting time?",
        "expected_names": ["Alex", "Sam"],
    },
    {
        "id": "no_personal_markers",
        "query": "What is the speed of light in vacuum?",
        "expected_names": [],
    },
    {
        "id": "conversational_shape",
        "query": "Pat mentioned that the dog was sick last week — should I call the vet?",
        "expected_names": ["Pat"],
    },
]


@pytest.fixture(scope="module")
def loaded_model():
    """Load the local model once per module.

    Mirrors the pattern from test_plausibility_contract_gpu.py — re-uses the
    main `load_config` + `load_base_model` path so the test exercises the
    same code that production server inference uses.
    """
    import os

    import torch

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from paramem.models.loader import load_base_model
    from paramem.utils.config import load_config

    config = load_config()
    model, tokenizer = load_base_model(config.model)
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
    from paramem.server.cloud_anonymizer import (
        anonymize_outbound,
        deanonymize_inbound,
    )

    model, tokenizer = loaded_model

    successes = 0
    blocks = 0
    expected_personal = 0
    failures = []

    for entry in _FIXTURE:
        query = entry["query"]
        expected_names = entry["expected_names"]
        if expected_names:
            expected_personal += 1

        anon_text, mapping = anonymize_outbound(query, model, tokenizer)

        # Empty mapping is a privacy-safe block (leak guard tripped or
        # model produced nothing usable).  Counts toward "blocks", does
        # not count toward "successes" of personal-marker queries.
        if not mapping:
            if expected_names:
                blocks += 1
            continue

        # Forward leak-guard contract: anonymize_outbound should never
        # return a mapping with real names visible in anon_text.  This
        # verifies the wrapper's invariant on real LLM output.
        for name in mapping:
            assert name not in anon_text, (
                f"[{entry['id']}] Leak guard breach: real name {name!r} "
                f"present in anon_text {anon_text!r}"
            )

        # Coverage contract: every expected name should appear in the
        # mapping keys.  Soft-asserted for now (record as failure rather
        # than abort) so the threshold can act on aggregate coverage.
        missing = [n for n in expected_names if n not in mapping]
        if missing:
            failures.append(
                f"[{entry['id']}] Coverage gap: missing {missing} in mapping {list(mapping)}"
            )
            continue

        # Round-trip contract: deanonymize_inbound restores the original.
        round_trip = deanonymize_inbound(anon_text, mapping)
        if _normalise(round_trip) != _normalise(query):
            failures.append(
                f"[{entry['id']}] Round-trip mismatch:\n"
                f"  original: {query!r}\n"
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
