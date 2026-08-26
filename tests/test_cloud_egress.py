"""Unit tests for the deanonymize side of the cloud egress chain
(``CloudScope`` observed scoping, binding-value pruning,
``deanonymize_facts``/``deanonymize_text``).

Pure-Python. These tests pin how ``AnonymizedContract`` is consumed;
production of it is out of scope here.
"""

from __future__ import annotations

from paramem.cloud.anonymize import AnonymizedContract
from paramem.cloud.deanonymize import CloudScope, deanonymize_facts, deanonymize_text


class TestCloudScopeObservedScoping:
    def _payload(self, reverse: dict[str, str], declared: frozenset[str]) -> AnonymizedContract:
        return AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            declared=declared,
            rekey_dropped=0,
            raw="",
        )

    def test_unobserved_token_is_absent_from_resolution(self):
        reverse = {"Person_1": "Alex", "Person_2": "Riley"}
        payload = self._payload(reverse, frozenset({"Person_1", "Person_2"}))
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_1 said hi",))
        assert scope.observed == frozenset({"Person_1"})
        assert "Person_1" in scope.resolution
        assert "Person_2" not in scope.resolution

    def test_declared_is_not_observed_scoped(self):
        reverse = {"Person_1": "Alex", "Person_2": "Riley"}
        payload = self._payload(reverse, frozenset({"Person_1", "Person_2"}))
        scope = CloudScope.response(payload, cloud_bindings=None, sent=("Person_1 said hi",))
        # declared holds BOTH tokens even though only Person_1 was observed.
        assert scope.declared == frozenset({"Person_1", "Person_2"})


class TestCloudScopeBindingValuePruning:
    """``CloudScope.response``: a binding whose own VALUE still carries an
    unresolvable placeholder token is pruned entirely, in ONE pass.
    """

    def _payload(self, reverse: dict[str, str], declared: frozenset[str]) -> AnonymizedContract:
        return AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            declared=declared,
            rekey_dropped=0,
            raw="",
        )

    def test_binding_value_with_unresolvable_placeholder_is_pruned(self):
        """``{"Role_1": "Senior Engineer at Org_9"}`` — Org_9 is never
        declared anywhere, so the WHOLE binding is dropped, not just its
        unresolvable fragment."""
        payload = self._payload({}, frozenset())
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
        payload = self._payload({"Org_9": "Acme"}, frozenset({"Org_9"}))
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
        payload = self._payload({}, frozenset())
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


class TestDeanonymizeResponseText:
    def _scope(self, reverse: dict[str, str], sent: tuple[str, ...]) -> CloudScope:
        payload = AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            declared=frozenset(reverse.keys()),
            rekey_dropped=0,
            raw="",
        )
        return CloudScope.response(payload, cloud_bindings=None, sent=sent)

    def test_observed_token_resolves(self):
        scope = self._scope({"Person_1": "Alex"}, sent=("Person_1",))
        assert deanonymize_text(scope, "Hello Person_1!") == "Hello Alex!"

    def test_declared_but_unobserved_token_drops_the_response(self):
        """The Person_N seeded for a name that never occurred in the turn
        must not resolve if it leaks into cloud prose.
        """
        scope = self._scope({"Person_1": "Alex", "Person_2": "Riley"}, sent=("Person_1",))
        # Person_2 was declared (seeded) but never shown to the cloud —
        # it must not be a rewrite rule, and its presence in cloud prose
        # must fail closed (drop), not resolve.
        assert deanonymize_text(scope, "Hello Person_2!") is None

    def test_no_placeholder_present_is_a_noop(self):
        scope = self._scope({"Person_1": "Alex"}, sent=("Person_1",))
        assert deanonymize_text(scope, "Hello there!") == "Hello there!"


class TestUnbypassableRawReverseMap:
    """Structural closure: there is no signature that accepts a bare
    reverse map for deanonymizing cloud text — only a CloudScope.
    """

    def test_deanonymize_text_requires_a_scope_object(self):
        import inspect

        params = inspect.signature(deanonymize_text).parameters
        assert list(params) == ["scope", "text"]
        assert params["scope"].annotation in ("CloudScope", CloudScope)
