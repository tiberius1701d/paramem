"""Unit tests for the ``pii_scope`` mechanism — the operator-controlled
entity-category MINT SELECTOR shipped at ``sanitization.cloud_scope`` and
threaded through both cloud paths (inference egress + consolidation
enrichment) via ``_build_anonymization_mapping``.

``pii_scope`` has exactly one job: it decides which entity TYPES get a
placeholder minted (and are therefore substituted out of the payload)
versus which travel to the cloud verbatim.  In-scope → blocked.
Out-of-scope → emitted.  There is a SINGLE default,
:data:`~paramem.graph.placeholders._DEFAULT_PII_SCOPE` (``{"person"}``) —
the value production actually ships at ``SanitizationConfig.cloud_scope``.

Why these tests exist: the scope knob is load-bearing for anonymization
coverage.  A regression that hardcodes a different default instead of
honouring the operator's choice would silently over-anonymize (destroying
cloud utility on places/orgs/etc.) or under-anonymize (leaking persons
when the operator widens the scope and the builder ignores them).  These
tests exercise the *mechanism* with several scope values as worked
examples — not just the production default.

Scope is the runtime parameter; the values used here are illustrative,
not exhaustive.  Adding a new internal type (``organization``,
``product``, …) should not require changing these tests — the ones that
filter by *type* already cover the new value via parametrisation.
"""

from __future__ import annotations

from paramem.graph.extractor import extract_and_anonymize_for_cloud
from paramem.graph.placeholders import _DEFAULT_PII_SCOPE

# ---------------------------------------------------------------------------
# extract_and_anonymize_for_cloud — empty-scope opt-out
# ---------------------------------------------------------------------------


def test_cloud_egress_empty_scope_returns_verbatim_passthrough():
    """``pii_scope=set()`` short-circuits before any LLM call.

    Distinguishes the operator opt-out path ``(transcript, {}, {})`` from
    the block path ``("", {}, {})``.  The caller (inference handler)
    forwards the verbatim transcript to the cloud unchanged when scope
    is empty; deanonymize_text is a no-op on an empty reverse map.
    """
    transcript = "[user] Alice told me about the Berlin trip.\n[assistant] Got it."
    # Model/tokenizer should NOT be invoked under the empty-scope path,
    # so passing ``None`` is safe — if the helper short-circuits as
    # designed, it never touches them.
    anon_text, mapping, reverse = extract_and_anonymize_for_cloud(
        transcript,
        model=None,  # type: ignore[arg-type]
        tokenizer=None,  # type: ignore[arg-type]
        pii_scope=set(),
    )
    assert anon_text == transcript  # verbatim passthrough
    assert mapping == {}  # no anonymization happened
    assert reverse == {}


def test_cloud_egress_empty_string_returns_block():
    """Empty input is the block sentinel ``("", {}, {})`` regardless of scope."""
    anon_text, mapping, reverse = extract_and_anonymize_for_cloud(
        "",
        model=None,  # type: ignore[arg-type]
        tokenizer=None,  # type: ignore[arg-type]
        pii_scope={"person"},
    )
    assert anon_text == ""
    assert mapping == {}
    assert reverse == {}


# ---------------------------------------------------------------------------
# Module-level constant invariant
# ---------------------------------------------------------------------------


def test_default_scope_is_a_frozenset():
    """The single module-wide scope default is a frozenset (immutable).

    A mutable default set would let a caller's accidental ``.add(...)``
    mutate every other caller's "default". Frozenset prevents that.
    """
    assert isinstance(_DEFAULT_PII_SCOPE, frozenset)


def test_default_scope_is_person_only():
    """``_DEFAULT_PII_SCOPE`` matches the value production actually ships
    (``SanitizationConfig.cloud_scope``, default ``["person"]``) — one
    constant, one policy, everywhere.

    Mutation: widen the default back to ``{"person", "place"}`` (or any
    other value production doesn't ship) -> this test fails.
    """
    assert _DEFAULT_PII_SCOPE == frozenset({"person"})


# ---------------------------------------------------------------------------
# ExtractionPipeline wiring — value flows from ExtractionConfig through
# ``ExtractionPipeline.kwargs`` to ``extract_graph``
# ---------------------------------------------------------------------------


def test_extraction_pipeline_kwargs_threads_pii_scope():
    """Value passed to ``ExtractionConfig.pii_scope`` appears in
    ``ExtractionPipeline.kwargs`` output under the ``pii_scope`` key.

    This closes the wiring-without-test gap: structural guards in
    ``test_extraction_pipeline_guard.py`` confirm the kwarg name lines
    up with ``extract_graph``'s signature, but don't verify the
    configured *value* actually flows.  Without this test, a regression
    that drops the ``pii_scope`` line from ``ExtractionConfig`` (or the
    corresponding ``pick("pii_scope", ...)`` entry in
    ``ExtractionPipeline.kwargs``) would compile and pass the structural
    guard, but silently revert consolidation to the primitive default
    scope.
    """
    from unittest.mock import MagicMock

    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

    pipeline = ExtractionPipeline(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=ExtractionConfig(pii_scope={"person", "place", "organization"}),
        prompts_dir=None,
    )

    # Config-level default propagates when no override is given.
    kwargs = pipeline.kwargs(speaker_id="speaker0")
    assert kwargs["pii_scope"] == {"person", "place", "organization"}

    # Per-call override wins over the config-level default.
    overridden = pipeline.kwargs(pii_scope={"person"}, speaker_id="speaker0")
    assert overridden["pii_scope"] == {"person"}

    # None at the config level is honoured (callers downstream interpret
    # this as "use the primitive default"). Distinct from set().
    pipeline.config.pii_scope = None
    kwargs = pipeline.kwargs(speaker_id="speaker0")
    assert kwargs["pii_scope"] is None
