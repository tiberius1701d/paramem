"""Unit tests for the ``pii_scope`` mechanism — the operator-controlled
NER/verify scope shipped at ``sanitization.cloud_scope`` and threaded
through both cloud paths (inference egress + consolidation enrichment).

Why these tests exist: the scope knob is now load-bearing for both
cloud paths.  A regression that hardcodes ``{person, place}`` instead
of honouring the operator's choice would silently over-anonymize
(destroying cloud utility on places/orgs/etc.) or under-anonymize
(leaking persons when the operator widens the scope and the filter
ignores them).  These tests exercise the *mechanism* with several
scope values as worked examples — not just the production default.

Scope is the runtime parameter; the values used here are illustrative,
not exhaustive.  Adding a new internal type (``organization``,
``product``, …) should not require changing these tests — the ones
that filter by *type* already cover the new value via parametrisation.
"""

from __future__ import annotations

import pytest

from paramem.graph.extractor import (
    _CLOUD_EGRESS_DEFAULT_SCOPE,
    _DEFAULT_PII_SCOPE,
    extract_and_anonymize_for_cloud,
    extract_pii_names_with_ner,
    verify_anonymization_completeness,
)
from paramem.graph.schema import Entity, Relation, SessionGraph

# ---------------------------------------------------------------------------
# extract_pii_names_with_ner — scope filter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _spacy_available():
    """Skip NER tests if spaCy / en_core_web_sm aren't loadable.

    spaCy is a soft dep — the helper degrades gracefully to an empty
    dict when the model isn't present.  Skip rather than fail so this
    module is portable to environments without the language model.
    """
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except Exception as exc:
        pytest.skip(f"spaCy / en_core_web_sm not available: {exc}")


# Mixed-category sentence covering PERSON, ORG, GPE in stable spaCy labels
# (verified manually with ``en_core_web_sm`` 3.8.x).  Adjust only if a
# spaCy upgrade renames a label — and re-run the probe before doing so.
_MIXED_TRANSCRIPT = "Alice works at Acme Corp in Berlin."
_PERSON_NAMES = {"Alice"}
_PLACE_NAMES = {"Berlin"}
_ORG_NAMES = {"Acme Corp"}


@pytest.mark.parametrize(
    "scope, expected_names, expected_types",
    [
        # Single-category scopes — only that category surfaces.
        pytest.param({"person"}, _PERSON_NAMES, {"person"}, id="person_only"),
        pytest.param({"place"}, _PLACE_NAMES, {"place"}, id="place_only"),
        pytest.param({"organization"}, _ORG_NAMES, {"organization"}, id="organization_only"),
        # Multi-category scopes — union surfaces.
        pytest.param(
            {"person", "place"},
            _PERSON_NAMES | _PLACE_NAMES,
            {"person", "place"},
            id="person_and_place",
        ),
        pytest.param(
            {"person", "place", "organization"},
            _PERSON_NAMES | _PLACE_NAMES | _ORG_NAMES,
            {"person", "place", "organization"},
            id="person_place_org",
        ),
        # Out-of-scope category that doesn't appear in the transcript —
        # nothing surfaces, no error.
        pytest.param({"event"}, set(), set(), id="event_no_matches"),
    ],
)
def test_extract_pii_names_with_ner_filters_by_scope(
    _spacy_available, scope, expected_names, expected_types
):
    """NER returns only names whose internal type is in ``pii_scope``.

    Parametrised with several scopes to prove the filter is scope-driven,
    not hardcoded to any specific subset.
    """
    result = extract_pii_names_with_ner(_MIXED_TRANSCRIPT, pii_scope=scope)
    assert set(result.keys()) == expected_names
    assert set(result.values()) == expected_types


def test_extract_pii_names_with_ner_empty_scope_returns_empty(_spacy_available):
    """Empty scope is the operator opt-out: no NER hits at all.

    Distinguished from ``pii_scope=None`` (which falls back to the
    primitive default).  ``set()`` means "operator turned this off".
    """
    result = extract_pii_names_with_ner(_MIXED_TRANSCRIPT, pii_scope=set())
    assert result == {}


def test_extract_pii_names_with_ner_none_falls_back_to_default(_spacy_available):
    """``pii_scope=None`` uses the primitive default ``{person, place}``."""
    result = extract_pii_names_with_ner(_MIXED_TRANSCRIPT, pii_scope=None)
    assert set(result.values()) <= _DEFAULT_PII_SCOPE
    # Default scope must include person+place; sanity check those surface
    # when the transcript carries them.
    assert "Alice" in result
    assert "Berlin" in result
    # Organization is out of the primitive default — must NOT surface.
    assert "Acme Corp" not in result


def test_extract_pii_names_with_ner_empty_transcript_returns_empty():
    """Empty input is handled before spaCy is even loaded.

    Whitespace-only behaviour is unspecified — it may early-return or
    run NER (which produces nothing on whitespace anyway).  The firm
    contract is empty-string → ``{}``.
    """
    assert extract_pii_names_with_ner("", pii_scope={"person"}) == {}


# ---------------------------------------------------------------------------
# verify_anonymization_completeness — scope filter
# ---------------------------------------------------------------------------


def _make_mixed_graph():
    """SessionGraph with a person, a place, and an organization.

    Used to verify that ``verify_anonymization_completeness`` flags
    only in-scope categories as potential leaks.  All three entity
    types appear both as standalone entities and as relation
    participants so the defensive relation-walk in the verify
    function picks them up too.
    """
    return SessionGraph(
        session_id="t",
        timestamp="2026-04-28T00:00:00Z",
        entities=[
            Entity(name="Alice", entity_type="person"),
            Entity(name="Berlin", entity_type="place"),
            Entity(name="Acme Corp", entity_type="organization"),
        ],
        relations=[
            Relation(
                subject="Alice",
                predicate="works_at",
                object="Acme Corp",
                relation_type="factual",
                confidence=1.0,
            ),
            Relation(
                subject="Alice",
                predicate="lives_in",
                object="Berlin",
                relation_type="factual",
                confidence=1.0,
            ),
        ],
    )


@pytest.mark.parametrize(
    "scope, anon_text, expected_leaked",
    [
        # Person-only scope: only person leaks flagged.
        pytest.param(
            {"person"},
            "Alice lives in Berlin and works at Acme Corp.",
            {"Alice"},
            id="person_scope_flags_person_only",
        ),
        # Person+place scope: both flagged.
        pytest.param(
            {"person", "place"},
            "Alice lives in Berlin and works at Acme Corp.",
            {"Alice", "Berlin"},
            id="person_place_scope_flags_both",
        ),
        # Organization scope: only org flagged.
        pytest.param(
            {"organization"},
            "Alice lives in Berlin and works at Acme Corp.",
            {"Acme Corp"},
            id="org_scope_flags_org_only",
        ),
        # Wider scope: every in-scope name flagged.
        pytest.param(
            {"person", "place", "organization"},
            "Alice lives in Berlin and works at Acme Corp.",
            {"Alice", "Berlin", "Acme Corp"},
            id="all_three_scope_flags_all",
        ),
    ],
)
def test_verify_anonymization_completeness_scope_drives_real_names(
    scope, anon_text, expected_leaked
):
    """``verify_anonymization_completeness`` only flags leaks whose type ∈ scope.

    Out-of-scope entities pass through unflagged — by design, so the
    operator's choice of ``cloud_scope`` controls which categories the
    cloud is allowed to see verbatim.
    """
    graph = _make_mixed_graph()
    # Empty mapping → every in-scope name is missing from the mapping
    # AND appears verbatim in anon_text (Case 1 leak in the verify docstring).
    leaked = verify_anonymization_completeness(
        graph, mapping={}, anon_facts=[], anon_transcript=anon_text, pii_scope=scope
    )
    assert set(leaked) == expected_leaked


def test_verify_anonymization_completeness_empty_scope_returns_empty():
    """Empty scope short-circuits to no leaks: nothing's in scope to flag.

    Operator's "off" signal — must not invent leaks where there's no
    policy asking us to look.
    """
    graph = _make_mixed_graph()
    leaked = verify_anonymization_completeness(
        graph,
        mapping={},
        anon_facts=[],
        anon_transcript="Alice lives in Berlin and works at Acme Corp.",
        pii_scope=set(),
    )
    assert leaked == []


def test_verify_anonymization_completeness_default_scope_is_person_place():
    """``pii_scope=None`` preserves the primitive default ``{person, place}``.

    Back-compat for callers that don't pass an explicit scope (e.g.
    legacy experiment scripts, the existing extraction-pipeline tests).
    """
    graph = _make_mixed_graph()
    leaked = verify_anonymization_completeness(
        graph,
        mapping={},
        anon_facts=[],
        anon_transcript="Alice lives in Berlin and works at Acme Corp.",
        pii_scope=None,
    )
    # Person + place must be flagged; organization must not.
    assert set(leaked) == {"Alice", "Berlin"}


# ---------------------------------------------------------------------------
# extract_and_anonymize_for_cloud — empty-scope opt-out
# ---------------------------------------------------------------------------


def test_cloud_egress_empty_scope_returns_verbatim_passthrough():
    """``pii_scope=set()`` short-circuits before any LLM call.

    Distinguishes the operator opt-out path ``(transcript, {})`` from
    the block path ``("", {})``.  The caller (inference handler)
    forwards the verbatim transcript to the cloud unchanged when scope
    is empty; deanonymize_text is a no-op on an empty mapping.
    """
    transcript = "[user] Alice told me about the Berlin trip.\n[assistant] Got it."
    # Model/tokenizer should NOT be invoked under the empty-scope path,
    # so passing ``None`` is safe — if the helper short-circuits as
    # designed, it never touches them.
    anon_text, mapping = extract_and_anonymize_for_cloud(
        transcript,
        model=None,  # type: ignore[arg-type]
        tokenizer=None,  # type: ignore[arg-type]
        pii_scope=set(),
    )
    assert anon_text == transcript  # verbatim passthrough
    assert mapping == {}  # no anonymization happened


def test_cloud_egress_empty_string_returns_block():
    """Empty input is the block sentinel ``("", {})`` regardless of scope."""
    anon_text, mapping = extract_and_anonymize_for_cloud(
        "",
        model=None,  # type: ignore[arg-type]
        tokenizer=None,  # type: ignore[arg-type]
        pii_scope={"person"},
    )
    assert anon_text == ""
    assert mapping == {}


# ---------------------------------------------------------------------------
# Module-level constant invariants
# ---------------------------------------------------------------------------


def test_default_scopes_are_frozensets():
    """Module-level scope defaults are frozensets (immutable).

    Mutable default sets would let a caller's accidental ``.add(...)``
    mutate every other caller's "default".  Frozenset prevents that.
    """
    assert isinstance(_DEFAULT_PII_SCOPE, frozenset)
    assert isinstance(_CLOUD_EGRESS_DEFAULT_SCOPE, frozenset)


def test_cloud_egress_default_is_subset_of_primitive_default():
    """Cloud-egress default must be no wider than the primitive default.

    Cloud egress ships a *narrower* policy than the primitives' fallback
    — narrower is fine, but a wider cloud-egress default would mean
    cloud egress flags categories the primitives' verify wouldn't, which
    would be confusing.  Lock the relationship in.
    """
    assert _CLOUD_EGRESS_DEFAULT_SCOPE <= _DEFAULT_PII_SCOPE
