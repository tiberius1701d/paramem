"""Unit tests for paramem.graph.entity_correction.correct_entity_surfaces.

All model calls are mocked (patch ``generate_answer``) — this is a pure
gate/scope/parse unit test suite exercising the two gather loci
(reverse-mapping placeholders, entity attributes) against the ONE
``_verdict`` primitive's uniform apply gate. See
``.agent/design-place-correction-20260704.md`` ("EXTENSION (s38)" +
"DECOMPOSITION") for the live-model evidence trail this stage is built on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.graph.entity_correction import correct_entity_surfaces
from paramem.graph.schema import Entity

_DEFAULT_SCOPE = {"place", "organization", "concept", "attributes"}


def _model_tokenizer():
    """Return a (model, tokenizer) pair sufficient for the module's call path.

    ``tokenizer.apply_chat_template`` must return a plain string;
    ``generate_answer`` is patched separately per test so the tokenizer/model
    identity doesn't matter beyond that.
    """
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted-prompt"
    model = MagicMock()
    return model, tokenizer


class TestScopeGating:
    def test_empty_correction_entity_types_returns_empty_no_calls(self, monkeypatch):
        """(h) An empty correction_entity_types set is a no-op: no model calls,
        empty result, reverse_mapping untouched."""
        called = MagicMock()
        monkeypatch.setattr("paramem.graph.entity_correction.generate_answer", called)
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Frankfrut"}

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=set(),
        )

        assert result == []
        assert reverse_mapping == {"City_1": "Frankfrut"}
        called.assert_not_called()

    def test_none_correction_entity_types_returns_empty_no_calls(self, monkeypatch):
        """(h) None is treated the same as empty — no implicit default scope."""
        called = MagicMock()
        monkeypatch.setattr("paramem.graph.entity_correction.generate_answer", called)
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Frankfrut"}

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=None,
        )

        assert result == []
        called.assert_not_called()

    def test_correctable_kinds_empty_after_intersection_returns_empty(self, monkeypatch):
        """(h) A knob containing only non-correctable members (e.g. "person",
        "attributes") intersects to empty against {place,organization,concept}
        -> the stage is off, even with "attributes" present."""
        called = MagicMock()
        monkeypatch.setattr("paramem.graph.entity_correction.generate_answer", called)
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Frankfrut"}
        entity = Entity(name="Speaker0", entity_type="person", attributes={"origin": "Frankfrut"})

        result = correct_entity_surfaces(
            reverse_mapping,
            [entity],
            model,
            tokenizer,
            correction_entity_types={"person", "attributes"},
        )

        assert result == []
        called.assert_not_called()
        assert reverse_mapping == {"City_1": "Frankfrut"}
        assert entity.attributes == {"origin": "Frankfrut"}

    def test_empty_reverse_mapping_and_no_entities_returns_empty(self, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr("paramem.graph.entity_correction.generate_answer", called)
        model, tokenizer = _model_tokenizer()

        result = correct_entity_surfaces(
            {},
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert result == []
        called.assert_not_called()


class TestPlaceholderLocus:
    def test_place_placeholder_applied(self, monkeypatch):
        """(a) A place placeholder whose verdict is a known correction is applied."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Frankfrut"}

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Frankfrut", "kind": "place", "corrected": "Frankfurt", '
                '"is_known_entity": true}'
            ),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["City_1"] == "Frankfurt"
        assert result == [
            {
                "locus": "placeholder",
                "placeholder": "City_1",
                "type": "place",
                "kind": "place",
                "before": "Frankfrut",
                "after": "Frankfurt",
            }
        ]

    def test_placeholder_verdict_kind_person_is_a_cross_check_rejection(self, monkeypatch):
        """(b) A placeholder gathered as place-eligible (by anonymizer prefix)
        whose model verdict comes back kind=person is REJECTED by the uniform
        gate — the model's own kind classification is an independent
        cross-check on top of the gather-time anonymizer type."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Angela Merkl"}

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Angela Merkl", "kind": "person", "corrected": "Angela Merkel", '
                '"is_known_entity": true}'
            ),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["City_1"] == "Angela Merkl"
        assert result == []

    def test_person_placeholder_never_gathered(self, monkeypatch):
        """Person_N is excluded at gather time (its anonymizer type "person"
        is never in correctable_kinds) — no model call is made for it at all."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {
            "Person_1": "Jonh Smiht",
            "City_1": "Frankfrut",
        }
        calls = []

        def _fake_generate_answer(model, tokenizer, formatted, **kwargs):
            calls.append(formatted)
            return (
                '{"input": "Frankfrut", "kind": "place", "corrected": "Frankfurt", '
                '"is_known_entity": true}'
            )

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer", _fake_generate_answer
        )

        correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["Person_1"] == "Jonh Smiht"
        assert len(calls) == 1, "Only City_1 should ever reach generate_answer"

    def test_fiction_is_known_entity_false_not_applied(self, monkeypatch):
        """(f) is_known_entity=false leaves the surface untouched even if the
        model returned a different 'corrected' value."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Grenholm"}

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Grenholm", "kind": "place", "corrected": "Grenholm", '
                '"is_known_entity": false}'
            ),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["City_1"] == "Grenholm"
        assert result == []

    def test_corrected_equal_to_value_is_a_noop(self, monkeypatch):
        """corrected == value never counts as an applied change, even when
        is_known_entity is true."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"Thing_1": "Pytorch"}

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Pytorch", "kind": "concept", "corrected": "Pytorch", '
                '"is_known_entity": true}'
            ),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["Thing_1"] == "Pytorch"
        assert result == []

    def test_product_kind_normalizes_to_concept_and_applies(self, monkeypatch):
        """(i) A model-returned kind of "product" normalizes to "concept" so
        the gate's vocabulary stays stable, and the correction applies."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"Thing_1": "Pyttorch"}

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Pyttorch", "kind": "product", "corrected": "Pytorch", '
                '"is_known_entity": true}'
            ),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["Thing_1"] == "Pytorch"
        assert result == [
            {
                "locus": "placeholder",
                "placeholder": "Thing_1",
                "type": "concept",
                "kind": "concept",
                "before": "Pyttorch",
                "after": "Pytorch",
            }
        ]

    def test_parse_failure_skips_target_without_raising(self, monkeypatch):
        """A malformed model response for one target is skipped (leaves that
        surface unchanged) and does not raise."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "not json at all, no braces"}

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: "not json at all, no braces",
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping["City_1"] == "not json at all, no braces"
        assert result == []

    def test_multiple_placeholder_targets_return_list_of_applied_changes(self, monkeypatch):
        """The returned list matches every applied change, in placeholder-
        iteration order, for multiple in-scope entries."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {
            "City_1": "Frankfrut",
            "Org_1": "Wobbleco",  # fictional — model rejects
        }

        responses = iter(
            [
                (
                    '{"input": "Frankfrut", "kind": "place", "corrected": "Frankfurt", '
                    '"is_known_entity": true}'
                ),
                (
                    '{"input": "Wobbleco", "kind": "organization", "corrected": "Wobbleco", '
                    '"is_known_entity": false}'
                ),
            ]
        )
        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: next(responses),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert reverse_mapping == {"City_1": "Frankfurt", "Org_1": "Wobbleco"}
        assert result == [
            {
                "locus": "placeholder",
                "placeholder": "City_1",
                "type": "place",
                "kind": "place",
                "before": "Frankfrut",
                "after": "Frankfurt",
            }
        ]

    def test_custom_scope_restricts_placeholder_selection(self, monkeypatch):
        """Passing an explicit scope narrower than the default restricts
        which placeholders get a model call at all."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {
            "City_1": "Frankfrut",
            "Org_1": "Novatek Systams",
        }
        calls = []

        def _fake_generate_answer(model, tokenizer, formatted, **kwargs):
            calls.append(formatted)
            return '{"input": "x", "kind": "place", "corrected": "y", "is_known_entity": true}'

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer", _fake_generate_answer
        )

        correct_entity_surfaces(
            reverse_mapping,
            [],
            model,
            tokenizer,
            correction_entity_types={"place"},
        )

        # Only one in-scope entry (City_1) — exactly one generate call.
        assert len(calls) == 1


class TestAttributeLocus:
    def test_attribute_place_value_applied(self, monkeypatch):
        """(c) An entity attribute holding a misspelled well-known place is
        corrected when "attributes" is in the knob."""
        model, tokenizer = _model_tokenizer()
        entity = Entity(
            name="Speaker0",
            entity_type="person",
            attributes={"current_location": "Frankfrut"},
        )

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Frankfrut", "kind": "place", "corrected": "Frankfurt", '
                '"is_known_entity": true}'
            ),
        )

        result = correct_entity_surfaces(
            {},
            [entity],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert entity.attributes == {"current_location": "Frankfurt"}
        assert result == [
            {
                "locus": "attribute",
                "entity": "Speaker0",
                "key": "current_location",
                "kind": "place",
                "before": "Frankfrut",
                "after": "Frankfurt",
            }
        ]

    def test_attribute_kind_person_not_applied(self, monkeypatch):
        """(d) An attribute value the model classifies as kind=person (e.g.
        last_name) is never applied — person is never in correctable_kinds."""
        model, tokenizer = _model_tokenizer()
        entity = Entity(
            name="Speaker0",
            entity_type="person",
            attributes={"last_name": "Merkl"},
        )

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "Merkl", "kind": "person", "corrected": "Merkel", '
                '"is_known_entity": true}'
            ),
        )

        result = correct_entity_surfaces(
            {},
            [entity],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert entity.attributes == {"last_name": "Merkl"}
        assert result == []

    def test_attribute_kind_other_not_applied(self, monkeypatch):
        """(e) An attribute value the model classifies as kind=other (e.g. an
        email address) is never applied."""
        model, tokenizer = _model_tokenizer()
        entity = Entity(
            name="Speaker0",
            entity_type="person",
            attributes={"email": "alex.morgan@example.com"},
        )

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: (
                '{"input": "alex.morgan@example.com", "kind": "other", '
                '"corrected": "alex.morgan@example.com", "is_known_entity": false}'
            ),
        )

        result = correct_entity_surfaces(
            {},
            [entity],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert entity.attributes == {"email": "alex.morgan@example.com"}
        assert result == []

    def test_attributes_not_in_knob_leaves_attribute_values_untouched(self, monkeypatch):
        """(g) When "attributes" is absent from the knob, the attribute locus
        is never gathered at all — no model call for attribute values, even
        though the knob still carries in-scope entity-type members."""
        model, tokenizer = _model_tokenizer()
        entity = Entity(
            name="Speaker0",
            entity_type="person",
            attributes={"current_location": "Frankfrut"},
        )
        calls = []

        def _fake_generate_answer(model, tokenizer, formatted, **kwargs):
            calls.append(formatted)
            return (
                '{"input": "Frankfrut", "kind": "place", "corrected": "Frankfurt", '
                '"is_known_entity": true}'
            )

        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer", _fake_generate_answer
        )

        result = correct_entity_surfaces(
            {},
            [entity],
            model,
            tokenizer,
            correction_entity_types={"place", "organization", "concept"},
        )

        assert result == []
        assert entity.attributes == {"current_location": "Frankfrut"}
        calls_assert_msg = "attribute locus must not be gathered without 'attributes' in scope"
        assert calls == [], calls_assert_msg

    def test_empty_and_whitespace_attribute_values_skipped(self, monkeypatch):
        """Empty/whitespace attribute values are never gathered as targets."""
        model, tokenizer = _model_tokenizer()
        entity = Entity(
            name="Speaker0",
            entity_type="person",
            attributes={"nickname": "", "origin": "   "},
        )
        called = MagicMock()
        monkeypatch.setattr("paramem.graph.entity_correction.generate_answer", called)

        result = correct_entity_surfaces(
            {},
            [entity],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        assert result == []
        called.assert_not_called()

    def test_both_loci_return_list_carries_locus_field(self, monkeypatch):
        """(j) The returned list carries "locus" for both the placeholder and
        attribute sources when both fire in the same call."""
        model, tokenizer = _model_tokenizer()
        reverse_mapping = {"City_1": "Frankfrut"}
        entity = Entity(
            name="Speaker0",
            entity_type="person",
            attributes={"current_location": "Novatek Systams"},
        )

        responses = iter(
            [
                (
                    '{"input": "Frankfrut", "kind": "place", "corrected": "Frankfurt", '
                    '"is_known_entity": true}'
                ),
                (
                    '{"input": "Novatek Systams", "kind": "organization", '
                    '"corrected": "Novatek Systems", "is_known_entity": true}'
                ),
            ]
        )
        monkeypatch.setattr(
            "paramem.graph.entity_correction.generate_answer",
            lambda *a, **kw: next(responses),
        )

        result = correct_entity_surfaces(
            reverse_mapping,
            [entity],
            model,
            tokenizer,
            correction_entity_types=_DEFAULT_SCOPE,
        )

        loci = {r["locus"] for r in result}
        assert loci == {"placeholder", "attribute"}
        assert reverse_mapping["City_1"] == "Frankfurt"
        assert entity.attributes["current_location"] == "Novatek Systems"
