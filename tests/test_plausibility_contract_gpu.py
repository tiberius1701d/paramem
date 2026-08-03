"""LLM-compliance contract test for the tuned plausibility prompt.

Why this test exists: the KEEP-by-default 6-rule plausibility prompt
(9d9ba20) was tuned on the live local judge against real sweep output.
Mocked unit tests cannot catch drift in LLM compliance when the prompt
is re-tuned; only a live model run can verify that the judge's keep/drop
decisions still match the ground-truth labels.

This test runs the real local model against a hand-curated fixture
(tests/fixtures/plausibility_contract.json) of 12 labeled facts (4 KEEP
+ 8 DROP) derived lexically from four of the six DROP rules — R1, R2,
R5, R6, two cases each. R3 has no cases; R4's cases were deliberately
removed (see the fixture's own ``rationale`` field — the extractor
provably never emits a conversation-role entity, so an R4 input is not
producible). It asserts the judge's keep/drop decisions match the
ground-truth labels on at least 75% of facts — the measured baseline
for Mistral 7B at temperature 0 against this fixture.

Why 75% and not 90%: Mistral 7B scores 9/12 against this fixture —
misses on one R1 case and both R6 cases (the R6 pair has never been
dropped in any measured run). ceil(12 * 0.75) = 9, so 9/12 is exactly
the knife-edge measured floor, not headroom. The KEEP-by-default bias
appears to override rule-match on edge cases regardless of added
imperative text. Pushing the threshold higher would either require
switching to a stronger judge (Claude, GPT) or a structural prompt
rework beyond adding "MUST drop" language. The 75% threshold preserves
the test as a genuine regression guard at the measured baseline.

Run: pytest tests/test_plausibility_contract_gpu.py --gpu

The server must be stopped so the model can claim the GPU. A module-
scoped fixture loads Mistral once and tears down on completion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU contract test requires --gpu flag",
    ),
]

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "plausibility_contract.json"
# See module docstring — 75% is the measured Mistral 7B baseline against
# this fixture at temperature 0. Raising it requires stronger judge or
# structural prompt rework, not just imperative language tweaks.
_MATCH_THRESHOLD = 0.75


@pytest.fixture()
def loaded_model(gpu_base_model):
    """Expose the session-scoped base model for plausibility contract tests.

    Delegates to the session-scoped ``gpu_base_model`` fixture
    (``conftest.py``) so the model is loaded exactly once per
    ``pytest --gpu`` invocation.  Inference-only: this file creates no
    adapters, so no teardown cleanup is required beyond what the session
    fixture handles.

    The 75% threshold above is anchored to Mistral 7B at temperature 0;
    loading any other model would silently re-calibrate against an
    untested baseline.
    """
    return gpu_base_model


@pytest.fixture(scope="module")
def fixture_data():
    with _FIXTURE_PATH.open() as f:
        return json.load(f)


def _fact_key(fact: dict) -> tuple[str, str, str]:
    return (fact["subject"], fact["predicate"], fact["object"])


def test_plausibility_prompt_llm_compliance(loaded_model, fixture_data):
    """Live local judge matches >= 75% of ground-truth keep/drop labels."""
    from paramem.graph.extractor import judge_plausibility

    model, tokenizer = loaded_model
    transcript = fixture_data["transcript"]
    labeled_facts = fixture_data["facts"]

    # Strip test-only fields before passing to the judge. The judge sees
    # only subject/predicate/object/relation_type/confidence — same as prod.
    judge_input = [
        {k: f[k] for k in ("subject", "predicate", "object", "relation_type", "confidence")}
        for f in labeled_facts
    ]

    # Mirror the production boundary (extraction_pipeline.py wraps every
    # extractor/judge call in base_model_inference): the judge must run on
    # the base model, never a personal adapter, so the result is not
    # contaminated by adapters that earlier shared-model tests left loaded.
    from paramem.models.loader import base_model_inference

    with base_model_inference(model):
        survivors, raw = judge_plausibility(judge_input, transcript, model, tokenizer)
    assert survivors is not None, f"Judge returned None (parse failure). Raw output: {raw!r}"

    survivor_keys = {_fact_key(f) for f in survivors}

    matches = []
    mismatches = []
    for fact in labeled_facts:
        key = _fact_key(fact)
        actual = "keep" if key in survivor_keys else "drop"
        expected = fact["expected"]
        record = {
            "id": fact["id"],
            "rule": fact.get("rule"),
            "expected": expected,
            "actual": actual,
            "note": fact.get("note", ""),
        }
        (matches if actual == expected else mismatches).append(record)

    n = len(labeled_facts)
    required = math.ceil(n * _MATCH_THRESHOLD)
    match_count = len(matches)

    if match_count < required:
        mismatch_report = "\n".join(
            f"  [{m['id']:>10}] rule={m['rule']!s:>5}  expected={m['expected']:>4}  "
            f"actual={m['actual']:>4}  -- {m['note']}"
            for m in mismatches
        )
        pytest.fail(
            f"Plausibility prompt compliance regression: {match_count}/{n} match "
            f"(need >= {required}, threshold {_MATCH_THRESHOLD:.0%}).\n"
            f"Mismatches:\n{mismatch_report}"
        )
