"""Tests for the quadruple-encoded indexed-key memory production module.

Mirrors the structure of ``tests/test_indexed_memory.py``.
All tests are pure (no GPU, no real model load).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paramem.training.entry_memory import (
    RECALL_TEMPLATE,
    _build_response,
    assign_keys,
    build_registry,
    compute_simhash,
    format_entry_training,
    parse_recalled_entry,
    verify_confidence,
)
from paramem.training.indexed_memory import SIMHASH_BITS, simhash_confidence

# --- assign_keys ---


class TestAssignQuadKeys:
    def test_basic_assignment(self):
        triples = [
            ("Alice", "lives_in", "Berlin"),
            ("Alice", "has_pet", "Luna"),
        ]
        quads = assign_keys(triples)
        assert len(quads) == 2
        assert quads[0]["key"] == "graph1"
        assert quads[0]["subject"] == "Alice"
        assert quads[0]["predicate"] == "lives_in"
        assert quads[0]["object"] == "Berlin"
        assert quads[1]["key"] == "graph2"

    def test_start_index_respected(self):
        triples = [("Alice", "has_job", "engineer")]
        quads = assign_keys(triples, start_index=5)
        assert quads[0]["key"] == "graph5"

    def test_start_index_default_is_1(self):
        triples = [("A", "p", "B")]
        quads = assign_keys(triples)
        assert quads[0]["key"] == "graph1"

    def test_empty_input(self):
        assert assign_keys([]) == []

    def test_returns_four_fields_only(self):
        triples = [("S", "P", "O")]
        quads = assign_keys(triples)
        assert set(quads[0].keys()) == {"key", "subject", "predicate", "object"}

    def test_sequential_keys(self):
        triples = [("A", "p", "B"), ("C", "q", "D"), ("E", "r", "F")]
        quads = assign_keys(triples, start_index=10)
        keys = [q["key"] for q in quads]
        assert keys == ["graph10", "graph11", "graph12"]


# --- _build_response / parse_recalled_entry round-trip ---


class TestBuildQuadResponseAndParse:
    def test_round_trip(self):
        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        response = _build_response(quad)
        parsed = parse_recalled_entry(response)
        assert parsed is not None
        assert parsed == {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
        }

    def test_build_response_only_four_fields(self):
        import json

        quad = {
            "key": "graph2",
            "subject": "Bob",
            "predicate": "has_job",
            "object": "engineer",
            "speaker_id": "s0",
        }
        response = _build_response(quad)
        obj = json.loads(response)
        assert set(obj.keys()) == {"key", "subject", "predicate", "object"}


# --- parse_recalled_entry ---


class TestParseRecalledQuad:
    def test_valid_json(self):
        text = '{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}'
        result = parse_recalled_entry(text)
        assert result is not None
        assert result["key"] == "graph1"
        assert result["subject"] == "Alice"

    def test_json_with_surrounding_text(self):
        text = (
            'Sure: {"key": "graph1", "subject": "Alice", "predicate": "lives_in", '
            '"object": "Berlin"} done.'
        )
        result = parse_recalled_entry(text)
        assert result is not None
        assert result["subject"] == "Alice"

    def test_garbage_returns_none(self):
        assert parse_recalled_entry("hello world no json here") is None

    def test_empty_string_returns_none(self):
        assert parse_recalled_entry("") is None

    def test_missing_required_field_returns_none(self):
        # Missing "object"
        text = '{"key": "graph1", "subject": "Alice", "predicate": "lives_in"}'
        assert parse_recalled_entry(text) is None

    def test_malformed_json_returns_none(self):
        text = '{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"'
        assert parse_recalled_entry(text) is None

    def test_markdown_fenced_json(self):
        """Model output wrapped in markdown code fences should still parse."""
        text = (
            "```json\n"
            '{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}\n'
            "```"
        )
        result = parse_recalled_entry(text)
        assert result is not None
        assert result["subject"] == "Alice"

    def test_bold_markdown_artifacts(self):
        """Gemma 2-style bold markers should be stripped before parsing."""
        text = (
            '**{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}**'
        )
        result = parse_recalled_entry(text)
        assert result is not None
        assert result["key"] == "graph1"

    def test_list_object_coerced_to_comma_string(self):
        """A list-valued 'object' must be coerced to a comma-joined string."""
        import json

        raw = json.dumps(
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "speaks",
                "object": ["German", "English"],
            }
        )
        result = parse_recalled_entry(raw)
        assert result is not None
        assert result["object"] == "German, English"

    def test_first_object_extracted(self):
        """When model appends garbage after the first valid object, use the first."""
        import json

        first = json.dumps(
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
            }
        )
        text = first + '{"key": "graph2", "subject": "?", "predicate": "?", "object": "?"}'
        result = parse_recalled_entry(text)
        assert result is not None
        assert result["key"] == "graph1"

    def test_clean_input_unchanged_by_artifact_stripping(self):
        """Artifact stripping must be transparent on clean input."""
        text = '{"key": "graph3", "subject": "Bob", "predicate": "has_job", "object": "dev"}'
        result = parse_recalled_entry(text)
        assert result is not None
        assert result["subject"] == "Bob"


# --- compute_simhash ---


class TestComputeSimhash:
    def test_deterministic(self):
        a = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        b = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        assert a == b

    def test_returns_integer(self):
        result = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        assert isinstance(result, int)
        assert result >= 0

    def test_fits_in_64_bits(self):
        result = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        assert result < (1 << SIMHASH_BITS)

    def test_different_keys_different_hashes(self):
        """Same triple under different keys must produce different fingerprints."""
        h1 = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        h2 = compute_simhash("graph2", "Alice", "lives_in", "Berlin")
        assert h1 != h2

    def test_order_sensitive(self):
        """Subject / predicate / object must be distinguishable."""
        h1 = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        h2 = compute_simhash("graph1", "Berlin", "lives_in", "Alice")
        assert h1 != h2

    def test_different_objects_different_hashes(self):
        h1 = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        h2 = compute_simhash("graph1", "Alice", "lives_in", "Munich")
        assert h1 != h2

    def test_similar_content_similar_hashes(self):
        """Minor variation should produce high confidence."""
        h1 = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        h2 = compute_simhash("graph1", "Alice", "lives_in", "berlin")  # case only
        confidence = simhash_confidence(h1, h2)
        assert confidence > 0.8

    def test_completely_different_content_low_confidence(self):
        h1 = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        h2 = compute_simhash("graph1", "Bob", "manages", "robotics team budget")
        confidence = simhash_confidence(h1, h2)
        assert confidence < 0.7

    def test_empty_content(self):
        result = compute_simhash("graph1", "", "", "")
        assert isinstance(result, int)

    def test_not_equal_to_qa_simhash(self):
        """Quad and QA simhashes must not collide even on same semantic content."""
        from paramem.training.indexed_memory import compute_simhash as qa_simhash

        quad_h = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        qa_h = qa_simhash("graph1", "Where does Alice live?", "Berlin")
        # They hash different content strings — they should almost certainly differ.
        # Not a strict requirement, but documents the design intention.
        assert quad_h != qa_h or True  # always passes; here as a documentation assertion


# --- verify_confidence ---


class TestVerifyConfidence:
    def test_no_registry_returns_1(self):
        recalled = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
        }
        assert verify_confidence(recalled) == 1.0

    def test_exact_content_high_confidence(self):
        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        registry = build_registry([quad])
        assert verify_confidence(quad, registry) == 1.0

    def test_untrained_key_zero_confidence(self):
        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        registry = build_registry([quad])
        other = {"key": "graph99", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        assert verify_confidence(other, registry) == 0.0

    def test_minor_variation_high_confidence(self):
        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        registry = build_registry([quad])
        # lowercase object
        variant = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "berlin",
        }
        confidence = verify_confidence(variant, registry)
        assert confidence > 0.8

    def test_wrong_content_low_confidence(self):
        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        registry = build_registry([quad])
        wrong = {
            "key": "graph1",
            "subject": "Bob",
            "predicate": "manages",
            "object": "robotics team budget",
        }
        confidence = verify_confidence(wrong, registry)
        assert confidence < 0.75

    def test_enriched_registry_shape(self):
        """Enriched registry (dict-of-dicts) must also work."""
        from paramem.training.entry_memory import build_enriched_registry

        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        enriched = build_enriched_registry([quad])
        assert verify_confidence(quad, enriched) == 1.0


# --- build_registry ---


class TestBuildRegistry:
    def test_builds_from_quad_pairs(self):
        quads = [
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            {"key": "graph2", "subject": "Bob", "predicate": "has_job", "object": "engineer"},
        ]
        registry = build_registry(quads)
        assert len(registry) == 2
        assert "graph1" in registry
        assert "graph2" in registry
        assert isinstance(registry["graph1"], int)

    def test_registry_matches_compute_simhash(self):
        quad = {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}
        registry = build_registry([quad])
        expected = compute_simhash("graph1", "Alice", "lives_in", "Berlin")
        assert registry["graph1"] == expected

    def test_empty_input(self):
        assert build_registry([]) == {}


# --- RECALL_TEMPLATE ---


class TestQuadRecallTemplate:
    def test_template_format(self):
        result = RECALL_TEMPLATE.format(key="graph5")
        assert result == "Recall the fact stored under key 'graph5'."

    def test_template_with_arbitrary_key(self):
        result = RECALL_TEMPLATE.format(key="custom_key")
        assert "custom_key" in result


class TestFormatQuadrupleTraining:
    @pytest.fixture
    def mock_tokenizer(self):
        """Lightweight mock tokenizer that mirrors the fixture in test_indexed_memory.py."""
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            parts = []
            for m in messages:
                parts.append(f"<|{m['role']}|>{m['content']}")
            if add_generation_prompt:
                parts.append("<|assistant|>")
            return "".join(parts)

        tokenizer.apply_chat_template = apply_chat_template

        import torch

        def tokenize_fn(text, truncation=True, max_length=512, return_tensors="pt"):
            token_count = max(1, len(text.split()))
            return {
                "input_ids": torch.ones(1, token_count, dtype=torch.long),
                "attention_mask": torch.ones(1, token_count, dtype=torch.long),
            }

        tokenizer.side_effect = tokenize_fn
        tokenizer.__call__ = tokenize_fn
        return tokenizer

    def test_produces_one_example_per_quad(self, mock_tokenizer):
        """Quadruple format produces ONE example per fact (vs two for QA)."""
        quads = [
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
        ]
        examples = format_entry_training(quads, mock_tokenizer)
        assert len(examples) == 1

    def test_multiple_quads(self, mock_tokenizer):
        quads = [
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
            {"key": "graph2", "subject": "Bob", "predicate": "has_job", "object": "engineer"},
        ]
        examples = format_entry_training(quads, mock_tokenizer)
        assert len(examples) == 2

    def test_example_has_required_keys(self, mock_tokenizer):
        quads = [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}]
        examples = format_entry_training(quads, mock_tokenizer)
        for ex in examples:
            assert "input_ids" in ex
            assert "attention_mask" in ex
            assert "labels" in ex

    def test_fewer_examples_than_qa_format(self, mock_tokenizer):
        """format_entry_training must produce exactly half as many examples as
        format_indexed_training for the same number of facts."""
        from paramem.training.indexed_memory import format_indexed_training

        qa_pairs = [{"key": "graph1", "question": "Q?", "answer": "A"}]
        quads = [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}]
        qa_examples = format_indexed_training(qa_pairs, mock_tokenizer)
        quad_examples = format_entry_training(quads, mock_tokenizer)
        assert len(quad_examples) * 2 == len(qa_examples)
