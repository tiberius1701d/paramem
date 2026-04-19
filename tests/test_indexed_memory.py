"""Tests for indexed key memory functions."""

from unittest.mock import MagicMock

import pytest

from paramem.training.indexed_memory import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    RECALL_TEMPLATE,
    SIMHASH_BITS,
    assign_keys,
    build_registry,
    compute_simhash,
    format_indexed_training,
    parse_recalled_pair,
    probe_keys_grouped_by_adapter,
    simhash_confidence,
    validate_recall,
    verify_confidence,
)

# --- compute_simhash ---


class TestComputeSimhash:
    def test_deterministic(self):
        a = compute_simhash("graph1", "Q?", "A")
        b = compute_simhash("graph1", "Q?", "A")
        assert a == b

    def test_returns_integer(self):
        result = compute_simhash("graph1", "Q?", "A")
        assert isinstance(result, int)
        assert result >= 0

    def test_fits_in_64_bits(self):
        result = compute_simhash("graph1", "Where does Alex live?", "Heilbronn")
        assert result < (1 << SIMHASH_BITS)

    def test_different_keys_different_hashes(self):
        """Same content under different keys must produce different fingerprints."""
        h1 = compute_simhash("graph1", "Q?", "A")
        h2 = compute_simhash("graph2", "Q?", "A")
        assert h1 != h2

    def test_different_content_different_hashes(self):
        h1 = compute_simhash("graph1", "Q?", "Heilbronn")
        h2 = compute_simhash("graph1", "Q?", "Munich")
        assert h1 != h2

    def test_similar_content_similar_hashes(self):
        """Minor variation should produce high confidence, not zero."""
        h1 = compute_simhash("graph1", "Where does Alex live?", "Heilbronn")
        h2 = compute_simhash("graph1", "Where does Alex live?", "heilbronn")
        confidence = simhash_confidence(h1, h2)
        assert confidence > 0.8

    def test_completely_different_content_low_confidence(self):
        """Unrelated content should produce low confidence."""
        h1 = compute_simhash("graph1", "Where does Alex live?", "Heilbronn")
        h2 = compute_simhash("graph1", "What does Maria manage?", "robotics team budget")
        confidence = simhash_confidence(h1, h2)
        assert confidence < 0.7

    def test_empty_content(self):
        result = compute_simhash("graph1", "", "")
        assert isinstance(result, int)


# --- simhash_confidence ---


class TestSimhashConfidence:
    def test_identical_hashes(self):
        assert simhash_confidence(42, 42) == 1.0

    def test_completely_different(self):
        # All bits differ
        all_ones = (1 << SIMHASH_BITS) - 1
        assert simhash_confidence(0, all_ones) == 0.0

    def test_one_bit_different(self):
        confidence = simhash_confidence(0, 1)
        expected = 1.0 - (1 / SIMHASH_BITS)
        assert abs(confidence - expected) < 1e-10

    def test_symmetry(self):
        assert simhash_confidence(42, 99) == simhash_confidence(99, 42)

    def test_returns_float_between_0_and_1(self):
        confidence = simhash_confidence(12345, 67890)
        assert 0.0 <= confidence <= 1.0


# --- verify_confidence ---


class TestVerifyConfidence:
    def test_no_registry_returns_1(self):
        recalled = {"key": "graph1", "question": "Q?", "answer": "A"}
        assert verify_confidence(recalled) == 1.0

    def test_exact_content_high_confidence(self):
        registry = build_registry([{"key": "graph1", "question": "Q?", "answer": "A"}])
        recalled = {"key": "graph1", "question": "Q?", "answer": "A"}
        assert verify_confidence(recalled, registry) == 1.0

    def test_untrained_key_zero_confidence(self):
        registry = build_registry([{"key": "graph1", "question": "Q?", "answer": "A"}])
        recalled = {"key": "graph15", "question": "Q?", "answer": "A"}
        assert verify_confidence(recalled, registry) == 0.0

    def test_minor_variation_high_confidence(self):
        registry = build_registry(
            [{"key": "graph1", "question": "Where does Alex live?", "answer": "Heilbronn"}]
        )
        recalled = {
            "key": "graph1",
            "question": "Where does Alex live?",
            "answer": "heilbronn",
        }
        confidence = verify_confidence(recalled, registry)
        assert confidence > 0.8

    def test_hallucination_low_confidence(self):
        """Key echoed but content from a different trained key."""
        registry = build_registry(
            [
                {
                    "key": "graph10",
                    "question": "What does Maria manage?",
                    "answer": "robotics team budget",
                },
            ]
        )
        hallucinated = {
            "key": "graph15",
            "question": "What does Maria manage?",
            "answer": "robotics team budget",
        }
        confidence = verify_confidence(hallucinated, registry)
        assert confidence == 0.0  # Key not in registry

    def test_wrong_content_low_confidence(self):
        """Trained key, completely wrong content."""
        registry = build_registry(
            [{"key": "graph1", "question": "Where does Alex live?", "answer": "Heilbronn"}]
        )
        recalled = {
            "key": "graph1",
            "question": "What does Maria manage?",
            "answer": "robotics team budget",
        }
        confidence = verify_confidence(recalled, registry)
        assert confidence < DEFAULT_CONFIDENCE_THRESHOLD


# --- build_registry ---


class TestBuildRegistry:
    def test_builds_from_keyed_pairs(self):
        keyed_pairs = [
            {"key": "graph1", "question": "Q1?", "answer": "A1"},
            {"key": "graph2", "question": "Q2?", "answer": "A2"},
        ]
        registry = build_registry(keyed_pairs)
        assert len(registry) == 2
        assert "graph1" in registry
        assert "graph2" in registry
        assert isinstance(registry["graph1"], int)

    def test_registry_matches_compute_simhash(self):
        keyed_pairs = [{"key": "graph1", "question": "Q?", "answer": "A"}]
        registry = build_registry(keyed_pairs)
        expected = compute_simhash("graph1", "Q?", "A")
        assert registry["graph1"] == expected

    def test_empty_input(self):
        assert build_registry([]) == {}


# --- assign_keys ---


class TestAssignKeys:
    def test_basic_assignment(self):
        qa_pairs = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
            {"question": "What pet does Alex have?", "answer": "Luna"},
        ]
        keyed = assign_keys(qa_pairs)
        assert len(keyed) == 2
        assert keyed[0]["key"] == "graph1"
        assert keyed[0]["question"] == "Where does Alex live?"
        assert keyed[0]["answer"] == "Heilbronn"
        assert keyed[1]["key"] == "graph2"

    def test_custom_start_index(self):
        qa_pairs = [{"question": "Q1?", "answer": "A1"}]
        keyed = assign_keys(qa_pairs, start_index=5)
        assert keyed[0]["key"] == "graph5"

    def test_empty_input(self):
        assert assign_keys([]) == []

    def test_preserves_all_fields(self):
        qa_pairs = [{"question": "Q?", "answer": "A", "extra": "ignored"}]
        keyed = assign_keys(qa_pairs)
        assert "extra" not in keyed[0]
        assert set(keyed[0].keys()) == {"key", "question", "answer"}


# --- parse_recalled_pair ---


class TestParseRecalledPair:
    def test_valid_json(self):
        text = '{"key": "graph1", "question": "Q?", "answer": "A"}'
        result = parse_recalled_pair(text)
        assert result["key"] == "graph1"
        assert result["question"] == "Q?"
        assert result["answer"] == "A"

    def test_json_with_surrounding_text(self):
        text = 'Here is the pair: {"key": "graph1", "question": "Q?", "answer": "A"} done.'
        result = parse_recalled_pair(text)
        assert result is not None
        assert result["question"] == "Q?"

    def test_garbage_returns_none(self):
        assert parse_recalled_pair("hello world no json here") is None

    def test_empty_string(self):
        assert parse_recalled_pair("") is None

    def test_missing_required_fields(self):
        text = '{"key": "graph1", "foo": "bar"}'
        assert parse_recalled_pair(text) is None

    def test_partial_fields(self):
        text = '{"question": "Q?", "answer": "A"}'
        result = parse_recalled_pair(text)
        assert result is not None
        assert result["question"] == "Q?"

    def test_malformed_json(self):
        text = '{"key": "graph1", "question": "Q?", "answer": "A"'
        assert parse_recalled_pair(text) is None

    def test_extracts_first_object_from_garbage_continuation(self):
        """Model generates correct JSON then garbage -- extract first object."""
        text = (
            '{"key": "graph1", "question": "Where does Alex live?", "answer": "Heilbronn"}'
            '{"key": "graph12", "question": "fake?", "answer": "garbage"}'
            "unicorn criptor"
        )
        result = parse_recalled_pair(text)
        assert result is not None
        assert result["key"] == "graph1"
        assert result["answer"] == "Heilbronn"

    def test_extracts_first_object_with_trailing_braces(self):
        text = '{"key": "graph1", "question": "Q?", "answer": "A"} extra } stuff'
        result = parse_recalled_pair(text)
        assert result is not None
        assert result["answer"] == "A"


# --- validate_recall ---


class TestValidateRecall:
    def test_exact_match_with_registry(self):
        keyed_pairs = [{"key": "graph1", "question": "Q?", "answer": "A"}]
        registry = build_registry(keyed_pairs)
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        recalled = {"key": "graph1", "question": "Q?", "answer": "A"}
        result = validate_recall(recalled, original, registry)
        assert result["exact_match"] is True
        assert result["confidence"] == 1.0

    def test_exact_match_without_registry(self):
        """Without registry, confidence defaults to 1.0."""
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        recalled = {"key": "graph1", "question": "Q?", "answer": "A"}
        result = validate_recall(recalled, original)
        assert result["exact_match"] is True
        assert result["confidence"] == 1.0

    def test_none_recalled(self):
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        result = validate_recall(None, original)
        assert result["exact_match"] is False
        assert result["confidence"] == 0.0
        assert result["recalled"] is None

    def test_key_mismatch_rejects(self):
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        recalled = {"key": "graph10", "question": "Q?", "answer": "A"}
        result = validate_recall(recalled, original)
        assert result["exact_match"] is False
        assert result["key_match"] is False

    def test_question_mismatch(self):
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        recalled = {"key": "graph1", "question": "Wrong?", "answer": "A"}
        result = validate_recall(recalled, original)
        assert result["exact_match"] is False
        assert result["question_match"] is False

    def test_answer_mismatch(self):
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        recalled = {"key": "graph1", "question": "Q?", "answer": "Wrong"}
        result = validate_recall(recalled, original)
        assert result["exact_match"] is False
        assert result["answer_match"] is False

    def test_whitespace_tolerance(self):
        original = {"key": "graph1", "question": "Q?", "answer": "A"}
        recalled = {"key": "graph1", "question": "Q? ", "answer": " A "}
        result = validate_recall(recalled, original)
        assert result["question_match"] is True
        assert result["answer_match"] is True

    def test_hallucination_caught_by_registry(self):
        """Hallucination: untrained key -- registry gives 0.0 confidence."""
        registry = build_registry([{"key": "graph10", "question": "Maria Q?", "answer": "budget"}])
        original = {"key": "graph15", "question": "N/A", "answer": "N/A"}
        hallucinated = {
            "key": "graph15",
            "question": "Maria Q?",
            "answer": "budget",
        }
        result = validate_recall(hallucinated, original, registry)
        assert result["confidence"] == 0.0
        assert result["exact_match"] is False


# --- format_indexed_training ---


class TestFormatIndexedTraining:
    @pytest.fixture
    def mock_tokenizer(self):
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
            token_count = len(text.split())
            return {
                "input_ids": torch.ones(1, token_count, dtype=torch.long),
                "attention_mask": torch.ones(1, token_count, dtype=torch.long),
            }

        tokenizer.side_effect = tokenize_fn
        tokenizer.__call__ = tokenize_fn
        return tokenizer

    def test_produces_two_examples_per_pair(self, mock_tokenizer):
        keyed_pairs = [
            {"key": "graph1", "question": "Q?", "answer": "A"},
        ]
        examples = format_indexed_training(keyed_pairs, mock_tokenizer)
        assert len(examples) == 2

    def test_multiple_pairs(self, mock_tokenizer):
        keyed_pairs = [
            {"key": "graph1", "question": "Q1?", "answer": "A1"},
            {"key": "graph2", "question": "Q2?", "answer": "A2"},
        ]
        examples = format_indexed_training(keyed_pairs, mock_tokenizer)
        assert len(examples) == 4

    def test_example_has_required_keys(self, mock_tokenizer):
        keyed_pairs = [{"key": "graph1", "question": "Q?", "answer": "A"}]
        examples = format_indexed_training(keyed_pairs, mock_tokenizer)
        for ex in examples:
            assert "input_ids" in ex
            assert "attention_mask" in ex
            assert "labels" in ex


# --- RECALL_TEMPLATE ---


class TestRecallTemplate:
    def test_template_format(self):
        result = RECALL_TEMPLATE.format(key="graph5")
        assert result == "Recall the QA pair stored under key 'graph5'."

    def test_template_with_arbitrary_key(self):
        result = RECALL_TEMPLATE.format(key="custom_key")
        assert "custom_key" in result


# ---------------------------------------------------------------------------
# Helpers for probe_keys_grouped_by_adapter tests
# ---------------------------------------------------------------------------


def _make_stub_model(adapter_names: list[str]):
    """Return a lightweight stub model with peft_config set to the given names."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}
    return model


def _make_stub_tokenizer():
    return MagicMock()


def _stub_probe_key(expected_results: dict):
    """Return a probe_key replacement that looks up results by key."""

    def _probe(model, tokenizer, key, **kwargs):
        return expected_results.get(key)

    return _probe


# ---------------------------------------------------------------------------
# probe_keys_grouped_by_adapter
# ---------------------------------------------------------------------------


class TestProbeKeysGroupedByAdapter:
    def test_minimises_switches(self, monkeypatch):
        """switch_adapter is called once per group, not once per key."""
        switch_calls = []

        def fake_switch(model, name):
            switch_calls.append(name)

        def fake_probe(model, tokenizer, key, **kwargs):
            return {"key": key, "answer": f"ans_{key}", "confidence": 1.0}

        monkeypatch.setattr(
            "paramem.training.indexed_memory.probe_key",
            fake_probe,
        )
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            fake_switch,
        )

        model = _make_stub_model(["procedural", "episodic", "episodic_interim_20260417T0000"])
        tokenizer = _make_stub_tokenizer()

        keys_by_adapter = {
            "procedural": ["p1", "p2"],
            "episodic": ["e1", "e2", "e3"],
            "episodic_interim_20260417T0000": ["s1"],
        }

        probe_keys_grouped_by_adapter(model, tokenizer, keys_by_adapter)

        assert switch_calls == [
            "procedural",
            "episodic",
            "episodic_interim_20260417T0000",
        ]

    def test_skips_unloaded_adapter(self, monkeypatch):
        """Unloaded adapter: warning emitted, no switch, keys map to None."""
        import logging

        switch_calls = []

        def fake_switch(model, name):
            switch_calls.append(name)

        def fake_probe(model, tokenizer, key, **kwargs):
            return {"key": key, "answer": f"ans_{key}", "confidence": 1.0}

        monkeypatch.setattr(
            "paramem.training.indexed_memory.probe_key",
            fake_probe,
        )
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            fake_switch,
        )

        # Attach a dedicated handler to capture the warning directly —
        # sidesteps differences between pytest's caplog, capsys, and
        # logging.lastResort behavior across environments.
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.WARNING)
        target = logging.getLogger("paramem.training.indexed_memory")
        prev_level = target.level
        target.addHandler(handler)
        target.setLevel(logging.WARNING)

        try:
            # Only "episodic" is loaded; interim adapter is absent.
            model = _make_stub_model(["episodic"])
            tokenizer = _make_stub_tokenizer()

            keys_by_adapter = {
                "episodic": ["e1"],
                "episodic_interim_20260417T0000": ["s1", "s2"],
            }

            results = probe_keys_grouped_by_adapter(model, tokenizer, keys_by_adapter)
        finally:
            target.removeHandler(handler)
            target.setLevel(prev_level)

        # Warning logged for the missing adapter.
        assert "episodic_interim_20260417T0000" in stream.getvalue()

        # No switch for the missing adapter; one switch for episodic.
        assert switch_calls == ["episodic"]
        # Missing keys map to None.
        assert results["s1"] is None
        assert results["s2"] is None
        # Loaded adapter still probed.
        assert results["e1"] is not None

    def test_results_match_per_key_probe(self, monkeypatch):
        """Grouped probe returns the same results as probing keys individually."""
        per_key_answers = {
            "k1": {"key": "k1", "answer": "A1", "confidence": 1.0},
            "k2": {"key": "k2", "answer": "A2", "confidence": 0.9},
            "k3": {"key": "k3", "answer": "A3", "confidence": 0.8},
        }

        def fake_switch(model, name):
            pass

        def fake_probe(model, tokenizer, key, **kwargs):
            return per_key_answers.get(key)

        monkeypatch.setattr(
            "paramem.training.indexed_memory.probe_key",
            fake_probe,
        )
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            fake_switch,
        )

        model = _make_stub_model(["episodic"])
        tokenizer = _make_stub_tokenizer()

        # Grouped call
        grouped = probe_keys_grouped_by_adapter(model, tokenizer, {"episodic": ["k1", "k2", "k3"]})

        # Per-key individual calls
        per_key = {k: per_key_answers.get(k) for k in ["k1", "k2", "k3"]}

        assert grouped == per_key

    def test_empty_dict(self, monkeypatch):
        """Empty input returns empty dict; switch_adapter never called."""
        switch_calls = []

        def fake_switch(model, name):
            switch_calls.append(name)

        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            fake_switch,
        )

        model = _make_stub_model(["episodic"])
        tokenizer = _make_stub_tokenizer()

        result = probe_keys_grouped_by_adapter(model, tokenizer, {})

        assert result == {}
        assert switch_calls == []

    def test_preserves_caller_order(self, monkeypatch):
        """switch_adapter is called in the order groups appear in the input dict."""
        switch_calls = []

        def fake_switch(model, name):
            switch_calls.append(name)

        def fake_probe(model, tokenizer, key, **kwargs):
            return {"key": key, "answer": "x", "confidence": 1.0}

        monkeypatch.setattr(
            "paramem.training.indexed_memory.probe_key",
            fake_probe,
        )
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            fake_switch,
        )

        model = _make_stub_model(["procedural", "episodic", "episodic_interim_20260417T0000"])
        tokenizer = _make_stub_tokenizer()

        keys_by_adapter = {
            "procedural": ["p1"],
            "episodic": ["e1"],
            "episodic_interim_20260417T0000": ["s1"],
        }

        probe_keys_grouped_by_adapter(model, tokenizer, keys_by_adapter)

        # Must match insertion order, not alphabetical order.
        assert switch_calls == [
            "procedural",
            "episodic",
            "episodic_interim_20260417T0000",
        ]
