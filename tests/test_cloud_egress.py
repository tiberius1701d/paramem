"""Unit tests for the deanonymize side of the cloud egress chain:
``CloudScope`` observed scoping (whole-word containment, so a shorter
declared token is not shown merely because a longer one that contains it
occurred), binding-value pruning, ``deanonymize_facts``, and
``deanonymize_text`` (the tolerant rendering walk plus its literal
fail-closed refusal).

Pure-Python. These tests pin how ``AnonymizedContract`` is consumed;
production of it is out of scope here.
"""

from __future__ import annotations

import inspect

from paramem.cloud.anonymize import AnonymizedContract
from paramem.cloud.deanonymize import CloudScope, deanonymize_facts, deanonymize_text


def _payload(reverse: dict[str, str], declared: frozenset[str]) -> AnonymizedContract:
    """Build a minimal :class:`AnonymizedContract` carrying only the
    ``reverse``/``declared`` fields the tests in this module exercise —
    the one contract-fixture builder shared by every test class below.
    """
    return AnonymizedContract(
        status="ok",
        forward={v: k for k, v in reverse.items()},
        reverse=reverse,
        anon_transcript="",
        declared=declared,
        rekey_dropped=0,
        raw="",
    )


class TestCloudScopeBindingValuePruning:
    """``CloudScope.response``: a binding whose own VALUE still carries an
    unresolvable placeholder token is pruned entirely, in ONE pass.
    """

    def test_binding_value_with_unresolvable_placeholder_is_pruned(self):
        """``{"Role_1": "Senior Engineer at Org_9"}`` — Org_9 is never
        declared anywhere, so the WHOLE binding is dropped, not just its
        unresolvable fragment."""
        payload = _payload({}, frozenset())
        scope = CloudScope.response(
            payload,
            cloud_bindings={"Role_1": "Senior Engineer at Org_9"},
            sent=("some payload",),
        )
        assert "Role_1" not in scope.cloud_bindings
        assert "Role_1" not in scope.resolution

    def test_binding_value_resolvable_against_core_is_kept(self):
        """A binding value referencing a token that DOES resolve (via the
        CORE reverse map) is kept whole — pruning is targeted, not a
        blanket rejection of any value containing placeholder-shaped
        text."""
        payload = _payload({"Org_9": "Acme"}, frozenset({"Org_9"}))
        scope = CloudScope.response(
            payload,
            cloud_bindings={"Role_1": "Senior Engineer at Org_9"},
            sent=("some payload mentioning Org_9",),
        )
        assert scope.cloud_bindings["Role_1"] == "Senior Engineer at Org_9"
        assert scope.resolution["Role_1"] == "Senior Engineer at Org_9"

    def test_pruning_is_one_pass_not_a_fixpoint(self):
        """A chain of two bindings (Role_1's value references Role_2,
        which is itself unresolvable via Org_9) is only pruned ONE level —
        documented, deliberate behaviour (see the docstring), not a bug.
        Each binding's value is checked against the resolvability domain
        computed ONCE from the ORIGINAL, unpruned bindings — Role_2 is
        still a valid binding KEY at the moment Role_1 is checked, so
        Role_1 survives even though Role_2 itself gets pruned in the SAME
        pass (no second pass re-checks Role_1 against the now-smaller,
        post-pruning domain)."""
        payload = _payload({}, frozenset())
        scope = CloudScope.response(
            payload,
            cloud_bindings={
                "Role_1": "the person known as Role_2",
                "Role_2": "Senior Engineer at Org_9",
            },
            sent=("some payload",),
        )
        # Role_2's value carries an unresolvable Org_9 -> pruned.
        assert "Role_2" not in scope.cloud_bindings
        # Role_1's value names Role_2 — a valid binding key in the
        # ORIGINAL (pre-pruning) domain used for this single pass — so it
        # survives, even though Role_2 no longer resolves post-pruning.
        assert scope.cloud_bindings["Role_1"] == "the person known as Role_2"


class TestDeanonymizeFactsAlwaysSubstitutes:
    """``deanonymize_facts`` takes NO graph and mutates nothing, and always
    substitutes: there is no whole-delta accept/reject verdict.  The
    fail-closed residual sweep (surfaced as ``predicate_dropped`` /
    ``residual_dropped``) is what sheds an individual fact, and
    ``collisions`` is always an informational diagnostic. Diagnostics are
    the caller's business (see
    ``paramem.graph.extractor._record_binding_diagnostics``).
    """

    def test_orphan_token_dropped_via_residual_sweep(self):
        """An unresolvable token (never declared -> orphan) is dropped
        individually by the fail-closed residual sweep, not by rejecting
        the whole delta."""
        payload = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            declared=frozenset({"Person_1"}),
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_1",))
        facts = [
            {
                "subject": "Person_1",
                "predicate": "colleague_of",
                "object": "Person_9",  # never declared -> orphan
                "relation_type": "social",
                "confidence": 0.9,
            }
        ]
        result = deanonymize_facts(scope, facts)
        assert result.facts == []
        assert len(result.residual_dropped) == 1
        # No cloud_bindings on this scope -> the collision scan never ran.
        assert result.collisions == []

    def test_clean_delta_substitutes(self):
        payload = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            declared=frozenset({"Person_1"}),
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_1",))
        facts = [
            {
                "subject": "Person_1",
                "predicate": "likes",
                "object": "coffee",
                "relation_type": "preference",
                "confidence": 0.9,
            }
        ]
        result = deanonymize_facts(scope, facts)
        assert result.collisions == []
        assert result.facts == [
            {
                "subject": "Alex",
                "predicate": "likes",
                "object": "coffee",
                "relation_type": "preference",
                "confidence": 0.9,
            }
        ]

    def test_binding_collision_is_inert_fact_still_substitutes(self):
        """An ``observed``-scoped collision (cloud rebinding a token it was
        already shown as a CORE reference) surfaces as a ``collisions``
        entry — the diagnostic the caller writes to
        ``cloud_binding_collisions`` — but is otherwise INERT: CORE-LAST
        precedence resolves the fact via the CORE reverse map regardless,
        never rejecting anything.
        """
        payload = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            declared=frozenset({"Person_1"}),
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.response(
            payload,
            # cloud rebinds Person_1 — a token it WAS shown (``sent``).
            cloud_bindings={"Person_1": "someone else entirely"},
            sent=("Person_1",),
        )
        assert "Person_1" in scope.observed
        facts = [
            {
                "subject": "Person_1",
                "predicate": "likes",
                "object": "coffee",
                "relation_type": "preference",
                "confidence": 0.9,
            }
        ]
        result = deanonymize_facts(scope, facts)
        assert result.collisions == ["Person_1"]
        assert result.facts == [
            {
                "subject": "Alex",
                "predicate": "likes",
                "object": "coffee",
                "relation_type": "preference",
                "confidence": 0.9,
            }
        ]

    def test_no_collision_carries_empty_collisions(self):
        """The accepted-shape exit also carries ``collisions`` — it is the
        scan result, not a hardcoded ``[]``.

        Here cloud mints a binding for a token it was NEVER shown
        (``Org_9`` is not in ``observed``), which is the legitimate mint
        case: no collision, and the mint resolves.
        """
        payload = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            declared=frozenset({"Person_1"}),
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.response(payload, cloud_bindings={"Org_9": "Acme"}, sent=("Person_1",))
        facts = [
            {
                "subject": "Person_1",
                "predicate": "works_at",
                "object": "Org_9",
                "relation_type": "factual",
                "confidence": 0.9,
            }
        ]
        result = deanonymize_facts(scope, facts)
        assert result.collisions == []
        assert result.facts[0]["subject"] == "Alex"
        assert result.facts[0]["object"] == "Acme"


class TestCloudScopeObservedWholeWordScoping:
    """``CloudScope.response``'s ``observed`` is whole-word containment
    over ``sent`` against the declared vocabulary — the tolerant restore
    in ``deanonymize_text`` requires this tightening: under substring
    containment, a shorter declared token would count as shown merely
    because a longer token containing it occurred.
    """

    def test_shorter_token_inside_a_longer_one_is_not_observed_or_resolved(self):
        payload = _payload(
            {"Person_1": "Alex", "Person_10": "Riley"},
            frozenset({"Person_1", "Person_10"}),
        )
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_10 called.",))
        assert "Person_10" in scope.observed
        assert "Person_1" not in scope.observed
        assert "Person_1" not in scope.resolution
        assert scope.resolution["Person_10"] == "Riley"

    def test_undeclared_placeholder_shaped_surface_in_sent_is_never_observed(self):
        """``observed`` is never a shape scrape: a placeholder-shaped
        surface in ``sent`` that was never declared for this contract
        (``Boeing_747``) does not enter ``observed``, even though
        ``Person_1`` — a declared token also present in the same string —
        does."""
        payload = _payload({"Person_1": "Alex"}, frozenset({"Person_1"}))
        scope = CloudScope.response(
            payload, cloud_bindings=None, sent=("Person_1 owns a Boeing_747",)
        )
        assert "Person_1" in scope.observed
        assert "Boeing_747" not in scope.observed


class TestDeanonymizeTextReplyRestoreAndRefusal:
    """``deanonymize_text`` — the prose exit gate: a rendering of a shown
    declared token restores to its real value; a declared token that
    survives literally (standalone, unshown, or glued into a longer
    identifier) refuses the whole reply.
    """

    def test_a_rendering_of_a_shown_token_restores(self):
        payload = _payload({"Person_1": "Alex"}, frozenset({"Person_1"}))
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_1 said hi.",))
        out = deanonymize_text(scope, "Hi PERSON_1, nice to meet you.")
        assert out == "Hi Alex, nice to meet you."

    def test_a_literal_declared_but_unshown_token_refuses(self):
        payload = _payload(
            {"Person_1": "Alex", "Person_9": "Riley"},
            frozenset({"Person_1", "Person_9"}),
        )
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_1 said hi.",))
        assert deanonymize_text(scope, "Person_9 is here too.") is None

    def test_a_glued_identifier_refuses(self):
        """The literal check reaches where the boundary-anchored rendering
        matcher deliberately does not: a declared token surviving inside a
        longer identifier is refused, not silently left in place."""
        payload = _payload({"Language_3": "Mandarin"}, frozenset({"Language_3"}))
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Language_3 mentioned.",))
        out = deanonymize_text(scope, "See language_proficiency_Language_3 for detail.")
        assert out is None

    def test_unshown_core_token_refuses_despite_a_rendering_cloud_binding(self):
        """A cloud-minted case-variant binding for an UNSHOWN core token
        (``PERSON_1``, never sent to cloud) is inert — it does not restore
        the reply's literal ``Person_1``, which surfaces as a declared
        token surviving literally and refuses the whole reply. (A
        lowercase binding key, e.g. ``person_1``, is filtered earlier by
        table normalize as placeholder-shape-ambiguous — unrelated to this
        invariant — so this test uses an uppercase rendering to reach
        :func:`~paramem.cloud.placeholders._resolution_map`.)"""
        payload = _payload({"Person_1": "Alex"}, frozenset({"Person_1"}))
        scope = CloudScope.response(
            payload, cloud_bindings={"PERSON_1": "the neighbour"}, sent=("some payload",)
        )
        assert "Person_1" not in scope.observed
        assert scope.resolution == {}
        assert deanonymize_text(scope, "Person_1 is here.") is None


class TestUnbypassableRawReverseMap:
    """``deanonymize_text``'s signature takes a ``CloudScope``, not a raw
    mapping — a bare ``reverse`` dict cannot be handed to it, so the
    ``observed``-scoped resolution is not bypassable at the call site."""

    def test_signature_takes_scope_and_text_not_a_raw_mapping(self):
        params = list(inspect.signature(deanonymize_text).parameters)
        assert params == ["scope", "text"]
