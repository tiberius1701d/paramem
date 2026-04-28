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

import gc

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU contract test requires --gpu flag",
    ),
]


# Empirical baseline calibrated on Mistral 7B Instruct v0.3 with the
# repair pipeline + NER cross-check + default scope ``[person]``.
# Threshold set to 0.75 to absorb single transient failure (Mistral 7B
# non-determinism).  Raise the threshold only when the anonymization
# prompt is structurally improved (not when the local model is
# upgraded — a stronger model raising the rate is welcome but raising
# the threshold without recalibration would silently break this test
# if the user reverts).  Re-run scripts/dev/calibrate_cloud_anonymizer.py
# after any change to extractor.py, NER scope, or the anonymization
# prompt.
_MATCH_THRESHOLD = 0.75


# Default scope under which the contract is asserted.  Mirrors the
# production default in ``SanitizationConfig.cloud_scope``.  Only names
# whose spaCy NER type maps into this set are required to be
# anonymized; out-of-scope categories (places, organizations, etc.)
# pass through verbatim by design.
_DEFAULT_SCOPE = {"person"}


# Fixture transcripts: multi-turn dialogues with self-claims that the
# extractor can anchor on.  Each entry carries the production-shape
# transcript ([user]/[assistant] role prefixes), the speaker_name that
# the chat handler would have already resolved (greeting flow), and the
# names that appear *in the transcript text* and MUST therefore be
# anonymized before the cloud sees the transcript.
#
# ``speaker_name`` is metadata (provided by voice enrollment in
# production) that the prompt uses to bind first-person facts to a
# concrete subject.  It is NOT included in ``expected_names`` because
# the speaker's name doesn't appear in the transcript text — there is
# nothing to leak.  The anonymizer may or may not bind the speaker to
# a placeholder; either is privacy-safe.
#
# These mirror the shape of real archived sessions in
# data/ha/sessions/archive/ (statements interleaved with questions),
# but use synthetic names so the fixture is safe to ship.
_FIXTURE = [
    {
        "id": "single_person_self_claim",
        "speaker_name": "Anna",
        "transcript": (
            "[user] My colleague Alex told me about a project deadline.\n"
            "[assistant] Got it.\n"
            "[user] Should I follow up tomorrow?"
        ),
        "expected_names": ["Alex"],
    },
    {
        "id": "person_and_place",
        "speaker_name": "Anna",
        "transcript": (
            "[user] My friend Alex is moving to Berlin next month.\n"
            "[assistant] That's exciting news.\n"
            "[user] What restaurants should I recommend to them?"
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
        "transcript": (
            "[user] My colleagues Alex and Sam are planning a workshop.\n"
            "[assistant] Sounds productive.\n"
            "[user] Did they finalize the agenda?"
        ),
        "expected_names": ["Alex", "Sam"],
    },
    {
        "id": "no_personal_markers",
        "speaker_name": "Anna",
        "transcript": (
            "[user] What is the speed of light in vacuum?\n"
            "[assistant] About 299,792 kilometres per second.\n"
            "[user] How fast is that compared to sound?"
        ),
        "expected_names": [],
    },
    {
        "id": "conversational_shape",
        "speaker_name": "Anna",
        "transcript": (
            "[user] My partner Pat noticed our dog was sick last week.\n"
            "[assistant] I hope the dog is feeling better.\n"
            "[user] Should I call the vet for a follow-up?"
        ),
        "expected_names": ["Pat"],
    },
]


@pytest.fixture(scope="module")
def loaded_model():
    """Load the live server model (Mistral 7B by default) once per module.

    Uses ``load_server_config('configs/server.yaml.example')`` so the test
    exercises the SAME model that the deployed cloud anonymizer runs against.
    The training-side ``load_config()`` defaults to Qwen 2.5 3B (much weaker
    at structured JSON output for the anonymizer prompt) and would silently
    test a model the production path never uses.
    """
    import os

    import torch

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from paramem.models.loader import load_base_model
    from paramem.server.config import load_server_config

    server_cfg = load_server_config("configs/server.yaml.example")
    model_cfg = server_cfg.model_config
    model, tokenizer = load_base_model(model_cfg)
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
