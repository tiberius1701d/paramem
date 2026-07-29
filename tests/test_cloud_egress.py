"""Unit tests for the unified anonymize -> cloud -> deanonymize chain.

Pure-Python — ``anonymize_transcript`` is mocked so no GPU is
required. These tests pin the chain's behaviour directly, independent of
any of the five migrated call sites.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from paramem.cloud.anonymize import (
    _MIN_ANONYMIZER_OUTPUT_TOKENS,
    AnonymizedContract,
    _render_anonymize_prompt,
    _slice_facts_to_envelope,
    anonymize,
    anonymize_transcript,
)
from paramem.cloud.deanonymize import CloudScope, deanonymize_facts, deanonymize_text
from paramem.cloud.placeholders import insert_placeholders
from paramem.graph.schema import Relation, SessionGraph, facts_from_relations
from paramem.utils.vram_guard import MIB_PER_TOKEN_TRANSIENT


def _stub_tokenizer() -> MagicMock:
    """A tokenizer stand-in whose ``apply_chat_template`` returns a fixed
    string, so the local fact-boundary packer (``_slice_facts_to_envelope``,
    exercised whenever ``transcript=""``) can render its own overhead probe
    without touching a real model. ``anonymize_transcript`` itself is
    mocked in every test in this module, so this tokenizer never reaches a
    real ``generate()`` call — only the packer's own rendering.
    """
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="rendered-prompt")
    return tok


class _CountingTokenizer:
    """Deterministic tokenizer double: one "token" per character.

    ``apply_chat_template`` concatenates every message's ``content`` — no
    chat-template markup — so the rendered prompt's length is exactly
    predictable from the template + JSON content, and
    ``estimate_tokens(text, tok) == len(text)`` exactly (via ``__call__``
    returning one id per character). This makes the packing/derivation
    arithmetic in ``_slice_facts_to_envelope`` / ``anonymize_transcript``
    testable against hand-computed boundaries instead of an opaque mock.

    The concatenated output also always contains
    ``supports_system_role``'s ``"SYSROLE_CHECK_MARKER"`` probe string
    (since concatenation never drops content), so
    :func:`~paramem.models.loader.adapt_messages` never folds the system
    message into the user turn — the rendered prompt is exactly
    ``system_prompt + user_prompt``.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


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


def _anonymize(graph: SessionGraph, **kwargs):
    """Test helper over :func:`anonymize` — renders ``graph.relations`` to
    facts (interface narrowing: ``anonymize`` takes a fact list, never a
    graph) and supplies dummy prompt templates, since every test in this
    module mocks ``anonymize_transcript`` and never reads them.
    """
    kwargs.setdefault("user_prompt_template", "")
    kwargs.setdefault("system_prompt", "")
    return anonymize(facts_from_relations(graph.relations), **kwargs)


class TestAnonymizeForCloudTokenEnvelopeDefault:
    """``anonymize`` and ``anonymize_transcript`` share ONE envelope
    default (U2/U3 — the flat ``max_tokens``/``_DEFAULT_ANONYMIZER_MAX_TOKENS``
    cap is retired). ``max_new_tokens`` is derived from this single
    ``token_envelope``, never a second, independently-configured knob.
    """

    def test_anonymize_default_matches_module_envelope(self):
        import inspect

        from paramem.cloud.anonymize import _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE

        default = inspect.signature(anonymize).parameters["token_envelope"].default
        assert default == _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE

    def test_anonymize_transcript_default_matches_module_envelope(self):
        import inspect

        from paramem.cloud.anonymize import (
            _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
            anonymize_transcript,
        )

        default = inspect.signature(anonymize_transcript).parameters["token_envelope"].default
        assert default == _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE

    def test_max_tokens_parameter_is_gone(self):
        """The flat cap is deleted, not just defaulted — a caller passing
        ``max_tokens=`` must get a clean TypeError, not silent
        acceptance."""
        import inspect

        assert "max_tokens" not in inspect.signature(anonymize).parameters
        from paramem.cloud.anonymize import anonymize_transcript

        assert "max_tokens" not in inspect.signature(anonymize_transcript).parameters


class TestAnonymizeForCloudOptOut:
    def test_empty_scrub_short_circuits_without_model_call(self):
        graph = _graph([_rel("Alex", "works_at", "Acme Corp")])
        with patch("paramem.cloud.anonymize.anonymize_transcript") as mocked:
            payload = _anonymize(
                graph, model=object(), tokenizer=object(), transcript="hello", scrub=set()
            )
        mocked.assert_not_called()
        assert payload.status == "opted_out"
        assert payload.forward == {}
        assert payload.reverse == {}
        assert payload.anon_transcript == "hello"
        # anon_facts is no longer a stored field — every production reader
        # derives it on demand via insert_placeholders(graph.relations,
        # payload.forward); an empty forward map is the identity
        # substitution (verbatim facts).
        assert insert_placeholders(facts_from_relations(graph.relations), payload.forward) == [
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
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=(None, "", "raw-output"),
        ):
            payload = _anonymize(
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
        # anon_facts is no longer a stored field. On a failed payload
        # ``forward`` is {} same as on the opted-out path, so an UNGATED
        # derivation via insert_placeholders would return graph.relations
        # verbatim (real names) — every production reader (enrich stage,
        # request_graph_enrichment, /calibrate/anonymize) gates on
        # ``status == "failed"`` and returns [] instead of deriving.
        assert payload.raw == "raw-output"


class TestAnonymizeForCloudOk:
    def test_ok_builds_forward_reverse_and_facts(self):
        graph = _graph([_rel("Alex", "works_at", "Acme Corp")])
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"Alex": "Person_1"}, "anon transcript", "raw"),
        ):
            payload = _anonymize(
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
        assert insert_placeholders(facts_from_relations(graph.relations), payload.forward) == [
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
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({}, "anon transcript", "raw"),
        ):
            payload = _anonymize(
                graph, model=object(), tokenizer=object(), transcript="hello", scrub={"person name"}
            )
        assert payload.status == "ok"
        assert payload.forward == {}
        assert payload.reverse == {}


class TestAnonymizeForCloudSpeakerValueGuard:
    def test_hostile_hint_never_creates_speaker_keyed_reverse_entry(self):
        """Speaker-value guard in ``_build_anonymization_mapping``
        (paramem/cloud/placeholders.py) is unbypassable — the ONLY route
        to ``reverse`` is that function.

        ``{"RealName": "speaker0"}`` never survives to
        ``_build_anonymization_mapping`` in the full chain — the shape
        normalizer (step 4) drops it first, since ``"speaker0"`` never
        matches the placeholder shape.  Belt-and-suspenders: two
        independent guards, neither reachable around.
        """
        graph = _graph([_rel("speaker0", "prefers", "coffee")])
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"RealName": "speaker0"}, "anon transcript", "raw"),
        ):
            payload = _anonymize(
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
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"speaker0": "Person_1"}, "anon transcript", "raw"),
        ):
            payload = _anonymize(
                graph, model=object(), tokenizer=object(), transcript="hi", scrub={"person name"}
            )
        assert "speaker0" not in payload.forward
        assert "Person_1" not in payload.reverse


class TestAnonymizeForCloudIdentityReconciliation:
    def test_rekey_onto_domain_surface_preserves_model_placeholder(self):
        graph = _graph([_rel("yang ming", "works_at", "acme")])
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"Yang Ming": "Person_1"}, "anon", "raw"),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
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
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"Someone Else": "Person_1"}, "anon", "raw"),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
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
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"yang ming": "person_1"}, "anon", "raw"),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                identity_domain=["yang ming", "acme"],
            )
        assert payload.norm_stats["dropped"] == 1
        assert payload.status == "failed"
        assert payload.failure == "guard"
        assert payload.forward == {}
        # anon_facts is no longer a stored field. The status-gated pattern
        # every production reader follows (never derive on a failed
        # payload) must yield [] here, never the real "yang ming" name —
        # that is exactly the leak this guard exists to prevent.
        gated_anon_facts = (
            []
            if payload.status == "failed"
            else insert_placeholders(facts_from_relations(graph.relations), payload.forward)
        )
        assert gated_anon_facts == []
        assert not any("yang ming" in str(v) for f in gated_anon_facts for v in f.values())

    def test_rekey_only_drop_still_fails_closed_symmetric_case(self):
        """Symmetric case to the normalize-drop regression above: an
        entry that SURVIVES normalize (shape-valid) but is dropped by
        rekey reconciliation alone (names nothing in this chunk) must
        still fail closed.
        """
        graph = _graph([_rel("yang ming", "works_at", "acme")])
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"Someone Not In Chunk": "Person_1"}, "anon", "raw"),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                identity_domain=["yang ming", "acme"],
            )
        assert payload.norm_stats["dropped"] == 0
        assert payload.rekey_dropped == 1
        assert payload.status == "failed"
        assert payload.failure == "guard"
        assert payload.forward == {}
        # anon_facts is no longer a stored field; see the status-gated
        # derivation note above.
        gated_anon_facts = (
            []
            if payload.status == "failed"
            else insert_placeholders(facts_from_relations(graph.relations), payload.forward)
        )
        assert gated_anon_facts == []

    def test_identity_domain_none_skips_reconciliation_and_guard(self):
        """Risk 3: an identity_domain=None case must never reconcile or guard."""
        graph = _graph([_rel("Yang Ming", "works_at", "Acme")])
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"Someone Else": "Person_1"}, "anon", "raw"),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
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
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({"Someone Trimmed Off": "Person_1"}, "anon", "raw"),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
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
    def _payload(self, reverse: dict[str, str], declared: frozenset[str]) -> AnonymizedContract:
        return AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            declared=declared,
            norm_stats={"inverted": 0, "dropped": 0},
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
    """``CloudScope.response`` (2026-07-22 cloud-admission redesign):
    a binding whose own VALUE still carries an unresolvable placeholder
    token is pruned entirely, ONE pass — replacing the fatal binding-value
    scan that used to live in ``_check_mapping_totality``.
    """

    def _payload(self, reverse: dict[str, str], declared: frozenset[str]) -> AnonymizedContract:
        return AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            declared=declared,
            norm_stats={"inverted": 0, "dropped": 0},
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
    """``deanonymize_facts`` takes NO graph and mutates nothing.  Always
    substitutes now (2026-07-22 cloud-admission redesign retired the
    whole-delta accept/reject ``verdict`` ``DeanonResult`` used to carry) —
    the fail-closed residual sweep (surfaced as ``predicate_dropped`` /
    ``residual_dropped``) is what still sheds an individual fact, and
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
            norm_stats={"inverted": 0, "dropped": 0},
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
            norm_stats={"inverted": 0, "dropped": 0},
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
        never rejecting anything. Inverts the pre-redesign expectation
        (used to reject the whole delta).
        """
        payload = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="",
            declared=frozenset({"Person_1"}),
            norm_stats={"inverted": 0, "dropped": 0},
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
            norm_stats={"inverted": 0, "dropped": 0},
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
            norm_stats={"inverted": 0, "dropped": 0},
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


# ---------------------------------------------------------------------------
# U2 — fact-boundary slicing (.agent/plan-anonymize-slicing.md §7 items 5-11)
# ---------------------------------------------------------------------------

# A fixed 3-fact fixture used across the slicing tests below. Real packing
# boundaries are hand-derived (via the module's own constants, never a
# hardcoded guess) using _CountingTokenizer's exact char-per-token measure —
# see the module docstring above for why that makes the arithmetic provable.
_SLICE_FACTS = [
    {"subject": "Alice", "predicate": "knows", "object": "Bob"},
    {"subject": "Carol", "predicate": "knows", "object": "Dave"},
    {"subject": "Erin", "predicate": "knows", "object": "Frank"},
]
_SLICE_TEMPLATE = "T{facts_json}"


class TestFactBoundarySlicing:
    """``_slice_facts_to_envelope`` — items 5-8, 11."""

    def test_facts_fitting_envelope_yield_one_slice(self):
        """Item 5: facts fitting the envelope -> exactly one slice, slice
        == all facts. Hand-derived: at envelope=500 the packer's reserve
        term is dominated by the ``_MIN_ANONYMIZER_OUTPUT_TOKENS`` (256)
        floor (m5 fix), and ``base + all-three-fragments + 256 <= 500``
        under ``_CountingTokenizer``'s exact char-per-token measure."""
        slices = _slice_facts_to_envelope(
            _SLICE_FACTS,
            _CountingTokenizer(),
            scrub={"person name"},
            user_prompt_template=_SLICE_TEMPLATE,
            system_prompt="",
            token_envelope=500,
        )
        assert slices == [_SLICE_FACTS]

    def test_facts_exceeding_envelope_split_preserving_order_no_split_fact(self):
        """Item 6: N > 1 slices; every fact appears in exactly one slice,
        order preserved, no fact split. Hand-derived boundary: at
        envelope=200, even a SINGLE fact's own cost — dominated by the
        ``_MIN_ANONYMIZER_OUTPUT_TOKENS`` (256) floor the packer reserves
        per slice (m5 fix) — already exceeds 200 on its own, so each fact
        lands in its own slice: exactly 3."""
        slices = _slice_facts_to_envelope(
            _SLICE_FACTS,
            _CountingTokenizer(),
            scrub={"person name"},
            user_prompt_template=_SLICE_TEMPLATE,
            system_prompt="",
            token_envelope=200,
        )
        assert len(slices) == 3
        # Order preserved, no fact split, every fact appears exactly once.
        flattened = [f for s in slices for f in s]
        assert flattened == _SLICE_FACTS

    def test_single_oversized_fact_gets_its_own_slice_no_crash(self):
        """Item 7: a single fact larger than the envelope is still
        emitted as its own slice — no crash, no infinite loop."""
        big_fact = {"subject": "Alice", "predicate": "knows", "object": "Bob"}
        slices = _slice_facts_to_envelope(
            [big_fact],
            _CountingTokenizer(),
            scrub={"person name"},
            user_prompt_template=_SLICE_TEMPLATE,
            system_prompt="",
            token_envelope=1,  # far below any real cost
        )
        assert slices == [[big_fact]]

    def test_empty_facts_returns_one_empty_slice(self):
        """Item 8: empty facts -> exactly one (empty) slice — the
        chat-egress shape, where the call must still run."""
        slices = _slice_facts_to_envelope(
            [],
            _CountingTokenizer(),
            scrub={"person name"},
            user_prompt_template=_SLICE_TEMPLATE,
            system_prompt="",
            token_envelope=8192,
        )
        assert slices == [[]]

    def test_compact_rendering_byte_identical_to_json_dumps_no_indent(self):
        """Item 11: the rendered facts JSON is byte-identical to
        ``json.dumps(facts)`` (compact, no ``indent=2`` artefact — no
        ``'\\n  "subject"'`` in the rendered prompt)."""
        rendered = _render_anonymize_prompt(
            _SLICE_FACTS,
            _CountingTokenizer(),
            scrub={"person name"},
            transcript="",
            user_prompt_template="{facts_json}",
            system_prompt="",
        )
        assert rendered == json.dumps(_SLICE_FACTS)
        assert '\n  "subject"' not in rendered
        assert "  " not in rendered  # no indent whitespace anywhere

    def test_real_packer_slice_never_overruns_the_envelope(self):
        """m5 missing test 2: for a slice produced by the REAL packer,
        ``prompt_tokens + max_new_tokens <= token_envelope`` — the
        property the whole packing change exists to enforce. Verified
        for EVERY slice a larger, more realistic fact set packs into,
        using ``_CountingTokenizer``'s exact char-per-token measure so
        both sides of the inequality are provable, not approximate."""
        tokenizer = _CountingTokenizer()
        template = "T{facts_json}"
        envelope = 600
        facts = [
            {"subject": f"Person{i}", "predicate": "knows", "object": f"Other{i}"}
            for i in range(12)
        ]

        slices = _slice_facts_to_envelope(
            facts,
            tokenizer,
            scrub={"person name"},
            user_prompt_template=template,
            system_prompt="",
            token_envelope=envelope,
        )
        assert len(slices) > 1, "fixture must actually exercise multi-slice packing"

        for slice_facts in slices:
            expected_prompt = _render_anonymize_prompt(
                slice_facts,
                tokenizer,
                scrub={"person name"},
                transcript="",
                user_prompt_template=template,
                system_prompt="",
            )
            prompt_tokens = len(expected_prompt)  # _CountingTokenizer: 1 tok/char

            captured = {}

            def _fake_generate(model, tok, formatted, *, max_new_tokens, temperature, seed):
                captured["max_new_tokens"] = max_new_tokens
                return json.dumps({"mapping": {}, "anonymized_transcript": []})

            with patch("paramem.cloud.anonymize.generate_answer", side_effect=_fake_generate):
                anonymize_transcript(
                    slice_facts,
                    model=object(),
                    tokenizer=tokenizer,
                    scrub={"person name"},
                    token_envelope=envelope,
                    user_prompt_template=template,
                    system_prompt="",
                )

            assert prompt_tokens + captured["max_new_tokens"] <= envelope, (
                f"slice with {len(slice_facts)} fact(s): prompt_tokens={prompt_tokens} + "
                f"max_new_tokens={captured['max_new_tokens']} > envelope={envelope}"
            )


class TestMaxNewTokensDerivation:
    """``anonymize_transcript`` — item 10: ``max_new_tokens`` handed to
    ``generate_answer`` equals ``envelope - measured prompt tokens``, and
    equals ``_MIN_ANONYMIZER_OUTPUT_TOKENS`` in the clamped case."""

    def test_max_new_tokens_equals_envelope_minus_prompt_tokens(self):
        tokenizer = _CountingTokenizer()
        template = "T{facts_json}"
        prompt_tokens = len(template.format(facts_json=json.dumps([])))
        envelope = prompt_tokens + 500  # comfortably above the reserve floor
        captured = {}

        def _fake_generate(model, tok, formatted, *, max_new_tokens, temperature, seed):
            captured["max_new_tokens"] = max_new_tokens
            return json.dumps({"mapping": {}, "anonymized_transcript": []})

        with patch("paramem.cloud.anonymize.generate_answer", side_effect=_fake_generate):
            anonymize_transcript(
                [],
                model=object(),
                tokenizer=tokenizer,
                scrub={"person name"},
                token_envelope=envelope,
                user_prompt_template=template,
                system_prompt="",
            )

        assert captured["max_new_tokens"] == envelope - prompt_tokens

    def test_max_new_tokens_clamped_to_floor_when_prompt_overruns_envelope(self):
        tokenizer = _CountingTokenizer()
        template = "T{facts_json}"
        prompt_tokens = len(template.format(facts_json=json.dumps([])))
        envelope = prompt_tokens - 10  # prompt alone already exceeds envelope
        captured = {}

        def _fake_generate(model, tok, formatted, *, max_new_tokens, temperature, seed):
            captured["max_new_tokens"] = max_new_tokens
            return json.dumps({"mapping": {}, "anonymized_transcript": []})

        with patch("paramem.cloud.anonymize.generate_answer", side_effect=_fake_generate):
            anonymize_transcript(
                [],
                model=object(),
                tokenizer=tokenizer,
                scrub={"person name"},
                token_envelope=envelope,
                user_prompt_template=template,
                system_prompt="",
            )

        assert captured["max_new_tokens"] == _MIN_ANONYMIZER_OUTPUT_TOKENS


class TestAtomicTranscriptSlicing:
    """``anonymize`` — item 9: a non-empty transcript is atomic (never
    sliced) regardless of how large ``facts`` is, plus the clamp WARNING
    when the prompt alone overruns the envelope."""

    def test_nonempty_transcript_makes_exactly_one_call_regardless_of_fact_count(self):
        graph = _graph([_rel(f"Person{i}", "knows", f"Other{i}") for i in range(20)])
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({}, "anon transcript", "raw"),
        ) as mocked:
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="a real transcript",
                scrub={"person name"},
            )
        mocked.assert_called_once()
        assert payload.slices == 1

    def test_clamp_warning_fires_when_prompt_alone_overruns_the_envelope(self, caplog):
        """A transcript-bearing call whose prompt alone leaves less than
        the output reserve proceeds with the clamped allowance and logs a
        WARNING naming the overshoot."""
        import logging

        tokenizer = _CountingTokenizer()
        template = "T{transcript}"
        long_transcript = "x" * 500
        raw = json.dumps({"mapping": {}, "anonymized_transcript": "unchanged"})

        caplog.set_level(logging.WARNING, logger="paramem.cloud.anonymize")
        with patch("paramem.cloud.anonymize.generate_answer", return_value=raw):
            anonymize_transcript(
                [],
                model=object(),
                tokenizer=tokenizer,
                scrub={"person name"},
                transcript=long_transcript,
                token_envelope=100,  # far smaller than the transcript alone
                user_prompt_template=template,
                system_prompt="",
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("clamped allowance" in r.getMessage() for r in warnings)


class TestDynamicVramClamp:
    """The dynamic VRAM clamp threaded through ``anonymize()`` (owner-
    approved 2026-07-28, live-fold evidence: a packer-correct 8,192-token
    call still faulted "device not ready" at 1,191 MiB free). One
    measurement at ``anonymize()`` entry via
    ``paramem.utils.vram_guard.effective_token_envelope``, threaded to
    both the fact packer and every slice's ``anonymize_transcript`` call.
    """

    def test_free_high_configured_envelope_reaches_anonymize_transcript(self):
        """Ample free VRAM: the effective envelope handed to
        ``anonymize_transcript`` equals the CONFIGURED value unchanged."""
        graph = _graph([_rel("Alice", "knows", "Bob")])
        captured = {}

        def _fake(*_args, **kwargs):
            captured["token_envelope"] = kwargs["token_envelope"]
            captured["configured_envelope"] = kwargs["configured_envelope"]
            return ({}, "", "raw")

        with (
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=_fake),
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.safe_empty_cache"),
            patch(
                "paramem.utils.vram_guard.torch.cuda.mem_get_info",
                return_value=(4096 * 2**20, 8192 * 2**20),
            ),
        ):
            _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                token_envelope=8192,
            )

        assert captured["token_envelope"] == 8192
        assert captured["configured_envelope"] == 8192

    def test_free_low_clamp_reaches_anonymize_transcript_below_ceiling(self):
        """Tight free VRAM (the live fault's 1,191 MiB reading): the
        effective envelope handed to ``anonymize_transcript`` is SMALLER
        than the configured ceiling, and the ceiling is still passed
        through separately so it can be named in a WARNING."""
        graph = _graph([_rel("Alice", "knows", "Bob")])
        captured = {}

        def _fake(*_args, **kwargs):
            captured["token_envelope"] = kwargs["token_envelope"]
            captured["configured_envelope"] = kwargs["configured_envelope"]
            return ({}, "", "raw")

        with (
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=_fake),
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.safe_empty_cache"),
            patch(
                "paramem.utils.vram_guard.torch.cuda.mem_get_info",
                return_value=(1191 * 2**20, 8192 * 2**20),
            ),
        ):
            _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                token_envelope=8192,
            )

        assert captured["configured_envelope"] == 8192
        assert captured["token_envelope"] == int(1191 / MIB_PER_TOKEN_TRANSIENT)
        assert captured["token_envelope"] < 8192

    def test_cpu_passthrough_no_cuda(self):
        """No CUDA available: the effective envelope equals the configured
        one and ``free_mib`` is ``None`` — CPU test suites never need a
        GPU to exercise ``anonymize()``."""
        graph = _graph([_rel("Alice", "knows", "Bob")])
        captured = {}

        def _fake(*_args, **kwargs):
            captured["token_envelope"] = kwargs["token_envelope"]
            captured["free_mib"] = kwargs["free_mib"]
            return ({}, "", "raw")

        with (
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=_fake),
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=False),
        ):
            _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                token_envelope=8192,
            )

        assert captured["token_envelope"] == 8192
        assert captured["free_mib"] is None

    def test_packer_and_derivation_end_to_end_respect_the_clamp(self):
        """Full ``anonymize()`` call, REAL packer + REAL max_new_tokens
        derivation (only ``generate_answer`` is mocked): a generous
        configured ceiling but tight free VRAM must still force the real
        packer to split into multiple slices, proving the effective
        envelope reaches BOTH consumers named in the design (the packer
        AND the per-slice derivation), not just one.
        """
        tokenizer = _CountingTokenizer()
        facts = [
            {"subject": f"Person{i}", "predicate": "knows", "object": f"Other{i}"}
            for i in range(30)
        ]
        template = "T{facts_json}"
        call_count = {"n": 0}

        def _fake_generate(model, tok, formatted, *, max_new_tokens, temperature, seed):
            call_count["n"] += 1
            assert max_new_tokens >= _MIN_ANONYMIZER_OUTPUT_TOKENS
            return json.dumps({"mapping": {}, "anonymized_transcript": []})

        with (
            patch("paramem.cloud.anonymize.generate_answer", side_effect=_fake_generate),
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.safe_empty_cache"),
            patch(
                "paramem.utils.vram_guard.torch.cuda.mem_get_info",
                # ~200 supportable tokens at MIB_PER_TOKEN_TRANSIENT — far
                # below what 30 facts need, so VRAM (not the configured
                # ceiling) is the binding constraint.
                return_value=(int(200 * MIB_PER_TOKEN_TRANSIENT * 2**20), 8192 * 2**20),
            ),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=tokenizer,
                transcript="",
                scrub={"person name"},
                token_envelope=100_000,  # generous ceiling — VRAM must bind, not config
                user_prompt_template=template,
                system_prompt="",
            )

        assert payload.status == "ok"
        assert payload.slices > 1
        assert call_count["n"] == payload.slices
        # No fact lost or duplicated across the VRAM-forced slicing.
        assert len(payload.facts) == len(facts)

    def test_atomic_path_warning_names_both_ceiling_and_clamped_value(self, caplog):
        """A transcript-bearing (atomic) call whose EFFECTIVE VRAM-clamped
        envelope is far below the configured ceiling logs a WARNING naming
        BOTH values distinctly, so an operator can tell "the ceiling was
        too small" apart from "live VRAM clamped it down"."""
        import logging

        tokenizer = _CountingTokenizer()
        template = "T{transcript}"
        transcript = "x" * 50
        raw = json.dumps({"mapping": {}, "anonymized_transcript": "unchanged"})

        caplog.set_level(logging.WARNING, logger="paramem.cloud.anonymize")
        with (
            patch("paramem.cloud.anonymize.generate_answer", return_value=raw),
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.safe_empty_cache"),
            patch(
                "paramem.utils.vram_guard.torch.cuda.mem_get_info",
                return_value=(1 * 2**20, 8192 * 2**20),  # a handful of tokens supportable
            ),
        ):
            graph = _graph([])
            _anonymize(
                graph,
                model=object(),
                tokenizer=tokenizer,
                transcript=transcript,
                scrub={"person name"},
                token_envelope=8192,
                user_prompt_template=template,
            )

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("effective envelope" in w and "operator ceiling 8192" in w for w in warnings), (
            warnings
        )


class TestWithinSliceCanonCollisionSurvives:
    """B1 regression guard (code review): the cross-KEY canon dedup in
    ``anonymize``'s merge loop must be scoped to entries carried over
    from a PRIOR slice only — never to two literal real-value keys that
    canonicalize identically but both arrive within the SAME call.

    ``_substitute_whole_words`` (the substitution primitive
    ``insert_placeholders`` uses) matches EXACTLY, so if the merge loop
    dropped the second of two canon-colliding keys, the dropped literal
    spelling would never be substituted and would egress the cloud
    payload UNMASKED. This is the transcript-bearing (single-slice,
    atomic) path — the one production shape a naive "always dedup on
    canonical key" fix would have silently broken.
    """

    def test_two_canon_colliding_keys_both_survive_with_distinct_placeholders(self):
        graph = _graph(
            [
                _rel("José García", "knows", "Riley"),
                _rel("Jose Garcia", "likes", "Coffee"),
            ]
        )
        # The model sees two literal spellings of the same real-world
        # entity (a diacritic-fold collision: canonical("José García") ==
        # canonical("Jose Garcia")) and — plausibly, since nothing in its
        # prompt tells it they're the same person — assigns them DIFFERENT
        # placeholders.
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=(
                {"José García": "Person_1", "Jose Garcia": "Person_2"},
                "anon transcript",
                "raw",
            ),
        ):
            payload = _anonymize(
                graph,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="a real transcript",
                scrub={"person name"},
            )

        assert payload.status == "ok"
        assert payload.slices == 1
        # Both literal spellings survive in `forward` — neither is
        # silently dropped by a canon-based dedup that (incorrectly)
        # applied within this single call.
        assert payload.forward == {"José García": "Person_1", "Jose Garcia": "Person_2"}
        # Distinct placeholders — the within-call duplicate-placeholder
        # closure (a SEPARATE mechanism, kept live/cumulative) is not
        # what's under test here, but pinning it costs nothing.
        assert len(set(payload.forward.values())) == 2


# ---------------------------------------------------------------------------
# U2 — cross-slice mapping merge (§7 items 12-15)
# ---------------------------------------------------------------------------


class TestCrossSliceMappingMerge:
    def test_same_entity_two_slices_first_wins_on_canonical_key(self):
        """Item 12: the same entity named in two slices with different
        placeholders -> first-wins on the canonical key; one entry in
        ``forward``."""
        facts = [
            {"subject": "Alex", "predicate": "knows", "object": "Riley"},
            {"subject": "Alex", "predicate": "likes", "object": "Coffee"},
        ]

        def _fake(slice_facts, *args, **kwargs):
            # Both slices independently name "Alex" — slice 1 mints
            # Person_1, slice 2 (a separate local call with no memory of
            # slice 1) independently mints Person_9 for the SAME real
            # value.
            if slice_facts[0]["object"] == "Riley":
                return ({"Alex": "Person_1"}, "", "raw1")
            return ({"Alex": "Person_9"}, "", "raw2")

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[[facts[0]], [facts[1]]],
            ),
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=_fake),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "ok"
        # First-wins: slice 1's Person_1 decision survives; slice 2's
        # independently-minted Person_9 for the same canonical real value
        # never enters the merged table.
        assert payload.forward == {"Alex": "Person_1"}
        assert len(payload.forward) == 1

    def test_two_distinct_entities_same_placeholder_renumbered(self):
        """Item 13: two DIFFERENT entities assigned ``Person_1`` in their
        respective slices -> the second is renumbered (``Person_2``);
        every value in the merged ``forward`` map is unique."""
        facts = [
            {"subject": "Alex", "predicate": "knows", "object": "Riley"},
            {"subject": "Jordan", "predicate": "knows", "object": "Casey"},
        ]
        call_count = 0

        def _fake_anonymize_transcript(slice_facts, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Slice 1 sees "Alex", slice 2 sees "Jordan" — both minted
            # Person_1 independently (as two separate local calls would).
            names = {f["subject"] for f in slice_facts} | {f["object"] for f in slice_facts}
            if "Alex" in names:
                return ({"Alex": "Person_1"}, "", "raw1")
            return ({"Jordan": "Person_1"}, "", "raw2")

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[[facts[0]], [facts[1]]],
            ),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_fake_anonymize_transcript,
            ),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "ok"
        assert len(payload.forward) == 2
        assert set(payload.forward.values()) == {"Person_1", "Person_2"}
        assert len(set(payload.forward.values())) == len(payload.forward)

    def test_renumber_preserves_multi_segment_prefix(self):
        """Item 14: the renumber picks the smallest free index per prefix
        and preserves a multi-segment prefix
        (``Home_Address_1`` -> ``Home_Address_2``)."""
        facts = [
            {"subject": "123 Main St", "predicate": "mentioned", "object": "x"},
            {"subject": "456 Oak Ave", "predicate": "mentioned", "object": "y"},
        ]

        def _fake_anonymize_transcript(slice_facts, *args, **kwargs):
            subj = slice_facts[0]["subject"]
            return ({subj: "Home_Address_1"}, "", "raw")

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[[facts[0]], [facts[1]]],
            ),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_fake_anonymize_transcript,
            ),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"physical address"},
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "ok"
        assert set(payload.forward.values()) == {"Home_Address_1", "Home_Address_2"}

    def test_placeholder_prefix_none_falls_back_without_raising(self):
        """Item 15: ``placeholder_prefix`` returns ``None`` for an
        unshaped token, and the exact fallback expression the per-slice
        merge loop uses (``placeholder_prefix(placeholder) or "Thing"``)
        substitutes ``"Thing"`` and ``mint_placeholder`` still mints a
        fresh, non-colliding token — no exception.

        (Reaching this fallback via a full ``anonymize()`` call is not
        possible in practice: an unshaped model placeholder is dropped by
        ``_normalize_anonymization_mapping`` before the merge loop ever
        sees it — every value that survives normalize already matches
        ``PLACEHOLDER_SHAPE_RE`` by construction. This test pins the
        fallback EXPRESSION itself, defensive code for a shape the
        merge's own callers cannot currently produce.)
        """
        from paramem.cloud.placeholders import mint_placeholder, placeholder_prefix

        unshaped = "notshaped"
        assert placeholder_prefix(unshaped) is None
        used = {"Person_1"}
        minted = mint_placeholder(used, placeholder_prefix(unshaped) or "Thing")
        assert minted == "Thing_1"


# ---------------------------------------------------------------------------
# U2 — per-slice fail-closed (§7 items 16-19)
# ---------------------------------------------------------------------------


class TestPerSliceFailClosed:
    def test_one_of_three_slices_parse_fails_others_survive(self):
        """Item 16: slice 2 of 3 parse-fails -> status="ok",
        slices_failed == 1, contract.facts excludes exactly slice 2's
        facts, includes slices 1 and 3."""
        facts = [
            {"subject": "Alex", "predicate": "knows", "object": "Riley"},
            {"subject": "Jordan", "predicate": "knows", "object": "Casey"},
            {"subject": "Sam", "predicate": "knows", "object": "Drew"},
        ]

        def _fake(slice_facts, *args, **kwargs):
            subj = slice_facts[0]["subject"]
            if subj == "Jordan":
                return (None, "", "raw-fail")
            return ({subj: "Person_1"}, "", "raw-ok")

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[[facts[0]], [facts[1]], [facts[2]]],
            ),
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=_fake),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "ok"
        assert payload.slices == 3
        assert payload.slices_failed == 1
        assert payload.facts == [facts[0], facts[2]]

    def test_all_slices_fail_yields_failed_status(self):
        """Item 17: all slices fail -> status="failed", facts == [],
        failure is "guard" when any guard fired, else "parse"."""
        facts = [
            {"subject": "Alex", "predicate": "knows", "object": "Riley"},
            {"subject": "Jordan", "predicate": "knows", "object": "Casey"},
        ]

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[[facts[0]], [facts[1]]],
            ),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                return_value=(None, "", "raw-fail"),
            ),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "failed"
        assert payload.facts == []
        assert payload.failure == "parse"

    def test_all_slices_fail_via_guard_reports_guard_failure(self):
        """Item 17 (guard variant): every slice fails via the
        domain-scoped guard -> failure == "guard"."""
        facts = [
            {"subject": "Someone Unmatched", "predicate": "knows", "object": "Nobody Else"},
        ]

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[facts],
            ),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                return_value=({"Someone Unmatched": "Person_1"}, "", "raw"),
            ),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                identity_domain=["a completely different node"],
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "failed"
        assert payload.failure == "guard"
        assert payload.facts == []

    def test_guard_does_not_fire_on_speaker_only_slice(self):
        """Item 18: a slice whose facts are all speaker-only endpoints
        does NOT fire the guard even when the model named something that
        failed reconciliation — the regression test for amendment point
        3, now at the per-slice level."""
        facts = [{"subject": "speaker0", "predicate": "likes", "object": "speaker1"}]

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[facts],
            ),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                return_value=({"Someone Trimmed Off": "Person_1"}, "", "raw"),
            ),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                identity_domain=["speaker0", "speaker1", "trimmed non-speaker node"],
                user_prompt_template="",
                system_prompt="",
            )

        assert payload.status == "ok"
        assert payload.slices_failed == 0

    def test_rekey_dropped_sums_across_slices_stays_per_entry(self):
        """Item 19: ``rekey_dropped`` sums across slices and stays
        per-entry (not per-slice)."""
        facts = [
            {"subject": "Alex", "predicate": "knows", "object": "known_node"},
            {"subject": "Jordan", "predicate": "knows", "object": "known_node"},
        ]

        def _fake(slice_facts, *args, **kwargs):
            subj = slice_facts[0]["subject"]
            # Both entries name something that will NOT reconcile onto
            # the domain (the domain only contains "known_node").
            return ({subj: "Person_1"}, "", "raw")

        with (
            patch(
                "paramem.cloud.anonymize._slice_facts_to_envelope",
                return_value=[[facts[0]], [facts[1]]],
            ),
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=_fake),
        ):
            payload = anonymize(
                facts,
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="",
                scrub={"person name"},
                identity_domain=["known_node"],
                user_prompt_template="",
                system_prompt="",
            )

        # Each slice's endpoints include "known_node" (a non-speaker
        # name), but the reconciled mapping is empty only when NOTHING
        # survives — here the guard fires per slice (each slice's sole
        # mapping entry fails to reconcile), so both slices fail closed
        # and rekey_dropped accumulates one drop per slice (2 total, not
        # 1 "per-slice" count).
        assert payload.rekey_dropped == 2


# ---------------------------------------------------------------------------
# D1 — anonymized_transcript validity rule matrix (§7 items 42-45)
# ---------------------------------------------------------------------------


class TestAnonymizedTranscriptValidityRuleMatrix:
    """The four-cell matrix of plan §5.2 / §1.4, directly against
    ``anonymize_transcript``."""

    _TEMPLATE_KWARGS = {
        "user_prompt_template": "{scrub_categories} {facts_json} {transcript}",
        "system_prompt": "system",
    }

    def _run(self, *, transcript: str, raw: str):
        tokenizer = _stub_tokenizer()
        with patch("paramem.cloud.anonymize.generate_answer", return_value=raw):
            return anonymize_transcript(
                [{"subject": "Alex", "predicate": "knows", "object": "Riley"}],
                model=object(),
                tokenizer=tokenizer,
                scrub={"person name"},
                transcript=transcript,
                **self._TEMPLATE_KWARGS,
            )

    def test_item_42_empty_transcript_any_mapping_missing_rewrite_is_ok(self):
        """``transcript=""`` + any mapping + missing/[] rewrite ->
        status="ok" (mapping returned, not None), anon_transcript=="",
        NOT fail-closed. Direct regression guard for the graph tier:
        mutation — restore the old empty-array check — makes every
        enrichment chunk fail closed, and this test fails."""
        raw = json.dumps({"mapping": {"Alex": "Person_1"}, "anonymized_transcript": []})
        mapping, anon_transcript, raw_output = self._run(transcript="", raw=raw)
        assert mapping == {"Alex": "Person_1"}
        assert anon_transcript == ""
        assert raw_output == raw

    def test_item_43_nonempty_transcript_empty_mapping_missing_rewrite_is_ok(self):
        """Non-empty transcript + ``mapping == {}`` + missing rewrite ->
        status="ok", empty forward/reverse (verified at the ``anonymize``
        level below), and the caller-visible ``anon_transcript`` is the
        ORIGINAL transcript verbatim — argument-sourced, never a model
        artefact (checked at the ``anonymize`` level since
        ``anonymize_transcript`` itself returns "" for the legitimate
        empty case; the argument fallback happens one level up)."""
        raw = json.dumps({"mapping": {}})
        mapping, anon_transcript, raw_output = self._run(
            transcript="[user] Alex knows Riley.", raw=raw
        )
        assert mapping == {}
        assert anon_transcript == ""
        assert raw_output == raw

        # Caller-visible outcome: anonymize() sources anon_transcript from
        # the ORIGINAL argument transcript, never a model artefact —
        # assert identity with the input string.
        original_transcript = "[user] Alex knows Riley."
        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=({}, "", raw),
        ):
            payload = anonymize(
                [{"subject": "Alex", "predicate": "knows", "object": "Riley"}],
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript=original_transcript,
                scrub={"person name"},
                user_prompt_template="",
                system_prompt="",
            )
        assert payload.status == "ok"
        assert payload.forward == {}
        assert payload.reverse == {}
        assert payload.anon_transcript is original_transcript

    def test_item_44_nonempty_transcript_nonempty_mapping_missing_rewrite_fails_closed(self):
        """Non-empty transcript + non-empty mapping + missing/empty
        rewrite -> status="failed", failure="parse", forward/reverse
        empty, anon_transcript=="", facts==[] (checked at the
        ``anonymize`` level)."""
        raw = json.dumps({"mapping": {"Alex": "Person_1"}})
        mapping, anon_transcript, raw_output = self._run(
            transcript="[user] Alex knows Riley.", raw=raw
        )
        assert mapping is None
        assert anon_transcript == ""
        assert raw_output == raw

        with patch(
            "paramem.cloud.anonymize.anonymize_transcript",
            return_value=(None, "", raw),
        ):
            payload = anonymize(
                [{"subject": "Alex", "predicate": "knows", "object": "Riley"}],
                model=object(),
                tokenizer=_stub_tokenizer(),
                transcript="[user] Alex knows Riley.",
                scrub={"person name"},
                user_prompt_template="",
                system_prompt="",
            )
        assert payload.status == "failed"
        assert payload.failure == "parse"
        assert payload.forward == {}
        assert payload.reverse == {}
        assert payload.anon_transcript == ""
        assert payload.facts == []

    def test_item_45_malformed_array_fails_closed_regardless_of_state(self):
        """A malformed array (a non-str element) stays fail-closed
        regardless of transcript/mapping state (shape check unchanged) —
        including over an EMPTY transcript, where every other malformed
        shape would otherwise be legitimate-empty."""
        raw = json.dumps(
            {
                "mapping": {"Alex": "Person_1"},
                "anonymized_transcript": ["ok turn", 42],
            }
        )
        mapping, anon_transcript, raw_output = self._run(transcript="", raw=raw)
        assert mapping is None
        assert anon_transcript == ""
        assert raw_output == raw
