"""Unit tests for the unified anonymize -> cloud -> deanonymize chain.

Pure-Python — ``anonymize_with_local_model`` is mocked so no GPU is
required. These tests pin the chain's behaviour directly, independent of
any of the five migrated call sites.
"""

from __future__ import annotations

from unittest.mock import patch

from paramem.graph.cloud_egress import (
    AnonymizedPayload,
    CloudScope,
    anonymize_for_cloud,
    deanonymize_facts,
    deanonymize_response_text,
)
from paramem.graph.schema import Relation, SessionGraph


def _graph(relations: list[Relation] | None = None, session_id: str = "s1") -> SessionGraph:
    return SessionGraph(
        session_id=session_id,
        timestamp="",
        entities=[],
        relations=relations or [],
    )


def _rel(subject: str, predicate: str, obj: str, speaker_id: str = "speaker0") -> Relation:
    return Relation(
        subject=subject,
        predicate=predicate,
        object=obj,
        relation_type="factual",
        speaker_id=speaker_id,
    )


class TestAnonymizeForCloudMaxTokensDefault:
    """``anonymize_for_cloud``'s only model call is the
    anonymizer, so its ``max_tokens`` default must match the anonymizer's
    own budget (``_DEFAULT_ANONYMIZER_MAX_TOKENS`` = 2048), not the
    graph-tier enrichment filter's larger budget
    (``_DEFAULT_FILTER_MAX_TOKENS`` = 8192) — a caller who omits the
    parameter must never get the wider budget.
    """

    def test_default_matches_anonymizer_budget(self):
        import inspect

        from paramem.graph.cloud_egress import (
            _DEFAULT_ANONYMIZER_MAX_TOKENS,
        )

        default = inspect.signature(anonymize_for_cloud).parameters["max_tokens"].default
        assert default == _DEFAULT_ANONYMIZER_MAX_TOKENS


class TestAnonymizeForCloudOptOut:
    def test_empty_scrub_short_circuits_without_model_call(self):
        graph = _graph([_rel("Alex", "works_at", "Acme Corp")])
        with patch("paramem.graph.cloud_egress.anonymize_with_local_model") as mocked:
            payload = anonymize_for_cloud(
                graph, model=object(), tokenizer=object(), transcript="hello", scrub=set()
            )
        mocked.assert_not_called()
        assert payload.status == "opted_out"
        assert payload.forward == {}
        assert payload.reverse == {}
        assert payload.anon_transcript == "hello"
        assert payload.anon_facts == [
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "Acme Corp",
                "relation_type": "factual",
                "confidence": 1.0,
            }
        ]


class TestAnonymizeForCloudFailClosed:
    def test_llm_mapping_none_yields_failed_status(self):
        graph = _graph([_rel("Alex", "works_at", "Acme Corp")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=(None, "", "raw-output"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="hello",
                scrub={"person name"},
            )
        assert payload.status == "failed"
        assert payload.forward == {}
        assert payload.reverse == {}
        assert payload.anon_transcript == ""
        assert payload.anon_facts == []
        assert payload.raw == "raw-output"


class TestAnonymizeForCloudOk:
    def test_ok_builds_forward_reverse_and_facts(self):
        graph = _graph([_rel("Alex", "works_at", "Acme Corp")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Alex": "Person_1"}, "anon transcript", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="hello",
                scrub={"person name"},
            )
        assert payload.status == "ok"
        assert payload.forward == {"Alex": "Person_1"}
        assert payload.reverse == {"Person_1": "Alex"}
        assert payload.anon_transcript == "anon transcript"
        assert payload.anon_facts == [
            {
                "subject": "Person_1",
                "predicate": "works_at",
                "object": "Acme Corp",
                "relation_type": "factual",
                "confidence": 1.0,
            }
        ]
        assert payload.declared == frozenset({"Person_1"})

    def test_empty_llm_mapping_is_ok_not_failed(self):
        """A legitimate 'ran, found nothing in scope' verdict is not a failure."""
        graph = _graph([_rel("Alex", "works_at", "Acme Corp")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({}, "anon transcript", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph, model=object(), tokenizer=object(), transcript="hello", scrub={"person name"}
            )
        assert payload.status == "ok"
        assert payload.forward == {}
        assert payload.reverse == {}


class TestAnonymizeForCloudSpeakerValueGuard:
    def test_hostile_hint_never_creates_speaker_keyed_reverse_entry(self):
        """Speaker-value guard in ``_build_anonymization_mapping``
        (paramem/graph/placeholders.py) is unbypassable — the ONLY route
        to ``reverse`` is that function.

        ``{"RealName": "speaker0"}`` never survives to
        ``_build_anonymization_mapping`` in the full chain — the shape
        normalizer (step 4) drops it first, since ``"speaker0"`` never
        matches the placeholder shape.  Belt-and-suspenders: two
        independent guards, neither reachable around.
        """
        graph = _graph([_rel("speaker0", "prefers", "coffee")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"RealName": "speaker0"}, "anon transcript", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph, model=object(), tokenizer=object(), transcript="hi", scrub={"person name"}
            )
        assert "speaker0" not in payload.reverse
        assert "speaker0" not in payload.reverse.values()

    def test_speaker_id_key_dropped_by_builder_when_shape_survives_normalize(self):
        """A hostile hint keyed on a speaker id, valued on a genuinely
        placeholder-shaped token, survives normalize (both sides
        plausible) and is caught by ``_build_anonymization_mapping``'s
        own key guard instead.
        """
        graph = _graph([_rel("speaker0", "prefers", "coffee")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"speaker0": "Person_1"}, "anon transcript", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph, model=object(), tokenizer=object(), transcript="hi", scrub={"person name"}
            )
        assert "speaker0" not in payload.forward
        assert "Person_1" not in payload.reverse


class TestAnonymizeForCloudIdentityReconciliation:
    def test_rekey_onto_domain_surface_preserves_model_placeholder(self):
        graph = _graph([_rel("yang ming", "works_at", "acme")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Yang Ming": "Person_1"}, "anon", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="",
                scrub={"person name"},
                identity_domain=["yang ming", "acme"],
            )
        assert payload.status == "ok"
        assert payload.forward == {"yang ming": "Person_1"}
        assert payload.rekey_dropped == 0

    def test_unmatched_entry_is_dropped_and_counted(self):
        graph = _graph([_rel("yang ming", "works_at", "acme")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Someone Else": "Person_1"}, "anon", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="",
                scrub={"person name"},
                identity_domain=["yang ming", "acme"],
            )
        # "Someone Else" reconciles to nothing -> rekey_dropped=1, but the
        # guard requires the RECONCILED mapping to be fully empty AND
        # relation endpoints to contain a non-speaker name to fail closed.
        assert payload.rekey_dropped == 1
        assert payload.status == "failed"

    def test_normalize_dropped_entry_still_fails_closed(self):
        """A raw model mapping that ``_normalize_anonymization_mapping``
        drops entirely (shape-invalid on BOTH sides — e.g. a 7B emitting
        consistently lowercase placeholders) must fail closed exactly
        like a rekey-only drop, never silently egress the real name
        verbatim.

        Reproduces the exact review repro: raw model mapping
        ``{"yang ming": "person_1"}`` — "person_1" fails
        PLACEHOLDER_SHAPE_RE (lowercase leading char) just as "yang ming"
        does, so the normalizer drops the pair as ambiguous
        (``norm_stats["dropped"] == 1``), leaving the post-normalize
        table empty BEFORE reconciliation ever runs.

        Mutation: derive the guard's ``reconciled_from_nonempty`` from the
        post-normalize table instead of the raw ``llm_mapping`` -> this
        test fails (status stays "ok", anon_facts carries the real name).
        """
        graph = _graph([_rel("yang ming", "works_at", "acme")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"yang ming": "person_1"}, "anon", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="",
                scrub={"person name"},
                identity_domain=["yang ming", "acme"],
            )
        assert payload.norm_stats["dropped"] == 1
        assert payload.status == "failed"
        assert payload.failure == "guard"
        assert payload.forward == {}
        assert payload.anon_facts == []
        assert not any("yang ming" in str(v) for f in payload.anon_facts for v in f.values())

    def test_rekey_only_drop_still_fails_closed_symmetric_case(self):
        """Symmetric case to the normalize-drop regression above: an
        entry that SURVIVES normalize (shape-valid) but is dropped by
        rekey reconciliation alone (names nothing in this chunk) must
        still fail closed.
        """
        graph = _graph([_rel("yang ming", "works_at", "acme")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Someone Not In Chunk": "Person_1"}, "anon", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="",
                scrub={"person name"},
                identity_domain=["yang ming", "acme"],
            )
        assert payload.norm_stats["dropped"] == 0
        assert payload.rekey_dropped == 1
        assert payload.status == "failed"
        assert payload.failure == "guard"
        assert payload.forward == {}
        assert payload.anon_facts == []

    def test_identity_domain_none_skips_reconciliation_and_guard(self):
        """Risk 3: an identity_domain=None case must never reconcile or guard."""
        graph = _graph([_rel("Yang Ming", "works_at", "Acme")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Someone Else": "Person_1"}, "anon", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="",
                scrub={"person name"},
                identity_domain=None,
            )
        assert payload.status == "ok"
        assert payload.forward == {"Someone Else": "Person_1"}
        assert payload.rekey_dropped == 0


class TestAnonymizeForCloudGuardDomainSeparation:
    """Guard-domain separation: the guard is derived from
    ``graph.relations``' endpoints, never from ``identity_domain``
    directly — this class pins the case that distinguishes the two.
    """

    def test_guard_does_not_fire_when_surviving_edges_are_speaker_only(self):
        """chunk_nodes (identity_domain) contains a trimmed-off non-speaker
        node, but every RELATION endpoint (the guard domain) is speaker-only.
        The guard must not fire even though the LLM named something that
        reconciles to nothing.
        """
        graph = _graph([_rel("speaker0", "likes", "speaker1")])
        with patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Someone Trimmed Off": "Person_1"}, "anon", "raw"),
        ):
            payload = anonymize_for_cloud(
                graph,
                model=object(),
                tokenizer=object(),
                transcript="",
                scrub={"person name"},
                # identity_domain (rekey domain) is LARGER than the
                # relation endpoints -- includes a non-speaker node with
                # no surviving edge in this chunk.
                identity_domain=["speaker0", "speaker1", "trimmed non-speaker node"],
            )
        assert payload.rekey_dropped == 1
        assert payload.status == "ok"


class TestCloudScopeObservedScoping:
    def _payload(self, reverse: dict[str, str], declared: frozenset[str]) -> AnonymizedPayload:
        return AnonymizedPayload(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            anon_facts=[],
            declared=declared,
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )

    def test_unobserved_token_is_absent_from_resolution(self):
        reverse = {"Person_1": "Alex", "Person_2": "Riley"}
        payload = self._payload(reverse, frozenset({"Person_1", "Person_2"}))
        scope = CloudScope.for_response(payload, sota_bindings=None, sent=("Person_1 said hi",))
        assert scope.observed == frozenset({"Person_1"})
        assert "Person_1" in scope.resolution
        assert "Person_2" not in scope.resolution

    def test_declared_is_not_observed_scoped(self):
        reverse = {"Person_1": "Alex", "Person_2": "Riley"}
        payload = self._payload(reverse, frozenset({"Person_1", "Person_2"}))
        scope = CloudScope.for_response(payload, sota_bindings=None, sent=("Person_1 said hi",))
        # declared holds BOTH tokens even though only Person_1 was observed.
        assert scope.declared == frozenset({"Person_1", "Person_2"})


class TestDeanonymizeFactsTotalityGate:
    """``deanonymize_facts`` takes NO graph and mutates nothing — both of
    the totality gate's findings (``verdict`` and ``collisions``) come
    back on ``DeanonResult``.  Diagnostics are the caller's business (see
    ``paramem.graph.extractor._record_binding_diagnostics``).
    """

    def test_orphan_token_rejects_whole_delta(self):
        payload = AnonymizedPayload(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            anon_facts=[],
            declared=frozenset({"Person_1"}),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.for_response(payload, sota_bindings=None, sent=("Person_1",))
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
        assert result.verdict, "expected a non-empty verdict"
        assert result.facts == []
        # No sota_bindings on this scope -> the collision scan never ran.
        assert result.collisions == []

    def test_clean_delta_substitutes(self):
        payload = AnonymizedPayload(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            anon_facts=[],
            declared=frozenset({"Person_1"}),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.for_response(payload, sota_bindings=None, sent=("Person_1",))
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
        assert result.verdict == []
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

    def test_collisions_surface_on_the_result_when_scoped(self):
        """An ``observed``-scoped collision (SOTA rebinding a token it was
        already shown as a CORE reference) surfaces BOTH as a rejection
        verdict and as a ``collisions`` entry — the second is what the
        caller writes to ``sota_binding_collisions``.

        Mutation: drop ``collisions`` from the rejected-exit
        ``DeanonResult`` (returning ``[]``) -> the caller stops recording
        the collision diagnostic -> this test fails.
        """
        payload = AnonymizedPayload(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            anon_facts=[],
            declared=frozenset({"Person_1"}),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.for_response(
            payload,
            # SOTA rebinds Person_1 — a token it WAS shown (``sent``).
            sota_bindings={"Person_1": "someone else entirely"},
            sent=("Person_1",),
        )
        assert "Person_1" in scope.observed
        result = deanonymize_facts(scope, [])
        assert result.verdict == ["Person_1"]
        assert result.collisions == ["Person_1"]
        assert result.facts == []

    def test_accepted_exit_carries_the_collision_scan_result(self):
        """The accepted exit also carries ``collisions`` — it is the scan
        result, not a hardcoded ``[]``.

        Through ``deanonymize_facts`` a NON-empty ``collisions`` implies a
        non-empty ``verdict``: ``CloudScope.observed`` is always a
        ``frozenset`` (never ``None``), and under the observed-scoped
        branch every collision is folded into the verdict.  So the
        accepted exit can only ever carry ``[]`` in production — the
        informational, not-folded collision shape exists only when
        ``_check_mapping_totality`` is called directly with
        ``observed=None`` (pinned in ``test_extraction_pipeline.py``).

        Here SOTA mints a binding for a token it was NEVER shown
        (``Org_9`` is not in ``observed``), which is the legitimate mint
        case: no collision, no verdict, and the mint resolves.
        """
        payload = AnonymizedPayload(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            anon_facts=[],
            declared=frozenset({"Person_1"}),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )
        scope = CloudScope.for_response(
            payload, sota_bindings={"Org_9": "Acme"}, sent=("Person_1",)
        )
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
        assert result.verdict == []
        assert result.collisions == []
        assert result.facts[0]["subject"] == "Alex"
        assert result.facts[0]["object"] == "Acme"


class TestDeanonymizeResponseText:
    def _scope(self, reverse: dict[str, str], sent: tuple[str, ...]) -> CloudScope:
        payload = AnonymizedPayload(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            anon_facts=[],
            declared=frozenset(reverse.keys()),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )
        return CloudScope.for_response(payload, sota_bindings=None, sent=sent)

    def test_observed_token_resolves(self):
        scope = self._scope({"Person_1": "Alex"}, sent=("Person_1",))
        assert deanonymize_response_text(scope, "Hello Person_1!") == "Hello Alex!"

    def test_declared_but_unobserved_token_drops_the_response(self):
        """The Person_N seeded for a name that never occurred in the turn
        must not resolve if it leaks into cloud prose.
        """
        scope = self._scope({"Person_1": "Alex", "Person_2": "Riley"}, sent=("Person_1",))
        # Person_2 was declared (seeded) but never shown to the cloud —
        # it must not be a rewrite rule, and its presence in cloud prose
        # must fail closed (drop), not resolve.
        assert deanonymize_response_text(scope, "Hello Person_2!") is None

    def test_no_placeholder_present_is_a_noop(self):
        scope = self._scope({"Person_1": "Alex"}, sent=("Person_1",))
        assert deanonymize_response_text(scope, "Hello there!") == "Hello there!"


class TestUnbypassableRawReverseMap:
    """Structural closure: there is no signature that accepts a bare
    reverse map for deanonymizing cloud text — only a CloudScope.
    """

    def test_deanonymize_response_text_requires_a_scope_object(self):
        import inspect

        params = inspect.signature(deanonymize_response_text).parameters
        assert list(params) == ["scope", "text"]
        assert params["scope"].annotation in ("CloudScope", CloudScope)
