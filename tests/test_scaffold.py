"""Unit tests for experiments/utils/scaffold.py.

Covers:
- V1/V2/V3 builders produce correct JSON shape (key, question, answer).
- All builders respect start_index and n_keys.
- No duplicate keys within a single build call.
- Deterministic output (same call → same result).
- VARIANT_BUILDERS dispatch matches builder functions.
- build_fill_keyed correct replacement and key preservation.
- build_fill_keyed raises ValueError on duplicate fill_indices.
- build_fill_keyed raises IndexError when real_qa_pool is too short.
"""

from __future__ import annotations

import pytest

from experiments.utils.scaffold import (
    PLACEHOLDER_STRINGS,
    V1,
    V2,
    V3,
    V3_EXTENDED,
    V4,
    V5,
    V6,
    V7,
    V8,
    VARIANT_BUILDERS,
    VARIANTS,
    build_fill_keyed,
    build_v1_scaffold,
    build_v2_scaffold,
    build_v3_scaffold,
    build_v4_scaffold,
    build_v5_scaffold,
    build_v6_scaffold,
    build_v7_scaffold,
    build_v8_scaffold,
)


class TestScaffoldConstants:
    def test_variants_tuple_contains_original_three(self):
        assert V1 in VARIANTS
        assert V2 in VARIANTS
        assert V3 in VARIANTS

    def test_variants_tuple_contains_extended_variants(self):
        assert V3_EXTENDED in VARIANTS
        assert V4 in VARIANTS
        assert V5 in VARIANTS
        assert V6 in VARIANTS
        assert V7 in VARIANTS
        assert V8 in VARIANTS

    def test_variants_tuple_has_nine_entries(self):
        # V1, V2, V3, V3_extended, V4, V5, V6, V7, V8
        assert len(VARIANTS) == 9

    def test_variant_builders_keys_match_variants(self):
        assert set(VARIANT_BUILDERS.keys()) == set(VARIANTS)

    def test_variant_builders_callables(self):
        for name, builder in VARIANT_BUILDERS.items():
            assert callable(builder), f"VARIANT_BUILDERS[{name!r}] is not callable"

    def test_placeholder_strings_nonempty(self):
        assert len(PLACEHOLDER_STRINGS) >= 1
        for s in PLACEHOLDER_STRINGS:
            assert isinstance(s, str) and len(s) > 0


class TestBuildV1Scaffold:
    def test_length(self):
        result = build_v1_scaffold(5)
        assert len(result) == 5

    def test_key_format(self):
        result = build_v1_scaffold(3)
        assert result[0]["key"] == "graph1"
        assert result[1]["key"] == "graph2"
        assert result[2]["key"] == "graph3"

    def test_question_format(self):
        result = build_v1_scaffold(3)
        assert result[0]["question"] == "TBD-Q-1"
        assert result[2]["question"] == "TBD-Q-3"

    def test_answer_format(self):
        result = build_v1_scaffold(3)
        assert result[0]["answer"] == "TBD-A-1"
        assert result[2]["answer"] == "TBD-A-3"

    def test_required_fields(self):
        for entry in build_v1_scaffold(10):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_no_duplicate_keys(self):
        result = build_v1_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_deterministic(self):
        assert build_v1_scaffold(10) == build_v1_scaffold(10)

    def test_start_index(self):
        result = build_v1_scaffold(3, start_index=5)
        assert result[0]["key"] == "graph5"
        assert result[0]["question"] == "TBD-Q-5"
        assert result[0]["answer"] == "TBD-A-5"
        assert result[2]["key"] == "graph7"

    def test_zero_keys(self):
        assert build_v1_scaffold(0) == []

    def test_single_key(self):
        result = build_v1_scaffold(1)
        assert len(result) == 1
        assert result[0]["key"] == "graph1"


class TestBuildV2Scaffold:
    def test_length(self):
        result = build_v2_scaffold(5)
        assert len(result) == 5

    def test_key_format(self):
        result = build_v2_scaffold(3)
        assert result[0]["key"] == "graph1"
        assert result[2]["key"] == "graph3"

    def test_question_format(self):
        result = build_v2_scaffold(3)
        assert result[0]["question"] == "Question for slot 1"
        assert result[2]["question"] == "Question for slot 3"

    def test_answer_format(self):
        # V2 answer is still a TBD placeholder (not structural Q)
        result = build_v2_scaffold(3)
        assert result[0]["answer"] == "TBD-A-1"
        assert result[2]["answer"] == "TBD-A-3"

    def test_required_fields(self):
        for entry in build_v2_scaffold(10):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_no_duplicate_keys(self):
        result = build_v2_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_deterministic(self):
        assert build_v2_scaffold(10) == build_v2_scaffold(10)

    def test_start_index(self):
        result = build_v2_scaffold(2, start_index=7)
        assert result[0]["key"] == "graph7"
        assert result[0]["question"] == "Question for slot 7"
        assert result[1]["key"] == "graph8"

    def test_zero_keys(self):
        assert build_v2_scaffold(0) == []

    def test_question_contains_placeholder_string(self):
        # "Question for slot" is a PLACEHOLDER_STRINGS entry
        result = build_v2_scaffold(1)
        assert any(ph in result[0]["question"] for ph in PLACEHOLDER_STRINGS)


class TestBuildV3Scaffold:
    def test_length(self):
        result = build_v3_scaffold(5)
        assert len(result) == 5

    def test_key_format(self):
        result = build_v3_scaffold(3)
        assert result[0]["key"] == "graph1"
        assert result[2]["key"] == "graph3"

    def test_question_is_pending(self):
        result = build_v3_scaffold(3)
        for entry in result:
            assert entry["question"] == "pending"

    def test_answer_is_pending(self):
        result = build_v3_scaffold(3)
        for entry in result:
            assert entry["answer"] == "pending"

    def test_required_fields(self):
        for entry in build_v3_scaffold(10):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_no_duplicate_keys(self):
        result = build_v3_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_deterministic(self):
        assert build_v3_scaffold(10) == build_v3_scaffold(10)

    def test_start_index(self):
        result = build_v3_scaffold(2, start_index=3)
        assert result[0]["key"] == "graph3"
        assert result[1]["key"] == "graph4"

    def test_zero_keys(self):
        assert build_v3_scaffold(0) == []

    def test_question_and_answer_contain_placeholder_string(self):
        result = build_v3_scaffold(1)
        assert any(ph in result[0]["question"] for ph in PLACEHOLDER_STRINGS)
        assert any(ph in result[0]["answer"] for ph in PLACEHOLDER_STRINGS)


class TestVariantBuildersDispatch:
    def test_v1_builder_matches_direct(self):
        assert VARIANT_BUILDERS[V1](5) == build_v1_scaffold(5)

    def test_v2_builder_matches_direct(self):
        assert VARIANT_BUILDERS[V2](5) == build_v2_scaffold(5)

    def test_v3_builder_matches_direct(self):
        assert VARIANT_BUILDERS[V3](5) == build_v3_scaffold(5)

    def test_all_variants_produce_same_key_structure(self):
        # All three variants must produce graph1..graphN keys identically.
        n = 5
        for variant in VARIANTS:
            result = VARIANT_BUILDERS[variant](n)
            expected_keys = [f"graph{i}" for i in range(1, n + 1)]
            assert [e["key"] for e in result] == expected_keys, (
                f"Variant {variant} key structure mismatch"
            )


class TestBuildFillKeyed:
    def _make_scaffold(self, n: int) -> list[dict]:
        return build_v1_scaffold(n)

    def _make_real_pool(self, n: int) -> list[dict]:
        return [{"question": f"What is {i}?", "answer": f"Answer {i}"} for i in range(n)]

    def test_basic_replacement(self):
        scaffold = self._make_scaffold(5)
        pool = self._make_real_pool(2)
        result = build_fill_keyed(scaffold, pool, fill_indices=[0, 2])
        assert len(result) == 2
        # Key comes from scaffold (slot 0 = graph1, slot 2 = graph3)
        assert result[0]["key"] == "graph1"
        assert result[1]["key"] == "graph3"
        # Q and A come from real pool
        assert result[0]["question"] == "What is 0?"
        assert result[0]["answer"] == "Answer 0"
        assert result[1]["question"] == "What is 1?"
        assert result[1]["answer"] == "Answer 1"

    def test_key_preserved_from_scaffold(self):
        scaffold = self._make_scaffold(10)
        pool = self._make_real_pool(3)
        result = build_fill_keyed(scaffold, pool, fill_indices=[4, 6, 9])
        assert result[0]["key"] == "graph5"
        assert result[1]["key"] == "graph7"
        assert result[2]["key"] == "graph10"

    def test_result_length_equals_fill_indices_length(self):
        scaffold = self._make_scaffold(10)
        pool = self._make_real_pool(5)
        result = build_fill_keyed(scaffold, pool, fill_indices=[0, 3, 7])
        assert len(result) == 3

    def test_empty_fill_indices(self):
        scaffold = self._make_scaffold(5)
        pool = self._make_real_pool(0)
        result = build_fill_keyed(scaffold, pool, fill_indices=[])
        assert result == []

    def test_duplicate_fill_indices_raises_value_error(self):
        scaffold = self._make_scaffold(5)
        pool = self._make_real_pool(3)
        with pytest.raises(ValueError, match="duplicates"):
            build_fill_keyed(scaffold, pool, fill_indices=[1, 1, 2])

    def test_pool_too_short_raises_index_error(self):
        scaffold = self._make_scaffold(5)
        pool = self._make_real_pool(1)  # only 1 entry
        with pytest.raises(IndexError):
            build_fill_keyed(scaffold, pool, fill_indices=[0, 1, 2])

    def test_fill_index_out_of_range_raises(self):
        scaffold = self._make_scaffold(3)
        pool = self._make_real_pool(1)
        with pytest.raises((IndexError, KeyError)):
            build_fill_keyed(scaffold, pool, fill_indices=[10])

    def test_result_fields_are_exactly_key_question_answer(self):
        scaffold = self._make_scaffold(5)
        pool = self._make_real_pool(2)
        result = build_fill_keyed(scaffold, pool, fill_indices=[0, 1])
        for entry in result:
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_deterministic(self):
        scaffold = self._make_scaffold(5)
        pool = self._make_real_pool(3)
        r1 = build_fill_keyed(scaffold, pool, fill_indices=[0, 2, 4])
        r2 = build_fill_keyed(scaffold, pool, fill_indices=[0, 2, 4])
        assert r1 == r2


# ---------------------------------------------------------------------------
# V4 scaffold
# ---------------------------------------------------------------------------


class TestBuildV4Scaffold:
    """V4 builder produces empty-Q/A entries (no per-slot content).

    The Q/A fields are required by downstream consumers (build_registry,
    format_indexed_training, validate_recall) so V4 keeps the fields but
    leaves them empty — semantically still "no per-slot content."
    """

    def test_length(self):
        assert len(build_v4_scaffold(5)) == 5

    def test_key_format(self):
        result = build_v4_scaffold(3)
        assert result[0]["key"] == "graph1"
        assert result[1]["key"] == "graph2"
        assert result[2]["key"] == "graph3"

    def test_question_field_present_and_empty(self):
        """V4 entries have a `question` field with an empty string."""
        for entry in build_v4_scaffold(5):
            assert entry.get("question") == "", f"V4 entry must have empty question: {entry}"

    def test_answer_field_present_and_empty(self):
        """V4 entries have an `answer` field with an empty string."""
        for entry in build_v4_scaffold(5):
            assert entry.get("answer") == "", f"V4 entry must have empty answer: {entry}"

    def test_field_set(self):
        """V4 entries have exactly key, question, answer fields."""
        for entry in build_v4_scaffold(10):
            assert set(entry.keys()) == {"key", "question", "answer"}, (
                f"V4 entry should have key/question/answer, got {set(entry.keys())}"
            )

    def test_no_duplicate_keys(self):
        result = build_v4_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_deterministic(self):
        assert build_v4_scaffold(10) == build_v4_scaffold(10)

    def test_start_index(self):
        result = build_v4_scaffold(3, start_index=7)
        assert result[0]["key"] == "graph7"
        assert result[1]["key"] == "graph8"
        assert result[2]["key"] == "graph9"

    def test_zero_keys(self):
        assert build_v4_scaffold(0) == []

    def test_single_key(self):
        result = build_v4_scaffold(1)
        assert len(result) == 1
        assert result[0] == {"key": "graph1", "question": "", "answer": ""}

    def test_variant_builders_dispatch(self):
        """VARIANT_BUILDERS[V4] dispatches to build_v4_scaffold."""
        assert VARIANT_BUILDERS[V4](5) == build_v4_scaffold(5)

    def test_compatible_with_build_registry(self):
        """V4 scaffold is consumable by build_registry without KeyError."""
        from paramem.training.indexed_memory import build_registry

        registry = build_registry(build_v4_scaffold(3))
        assert set(registry.keys()) == {"graph1", "graph2", "graph3"}


# ---------------------------------------------------------------------------
# V5 scaffold — uniform long natural-language template
# ---------------------------------------------------------------------------


class TestBuildV5Scaffold:
    """V5 builder produces uniform fluent-English Q/A across all slots."""

    def test_length(self):
        assert len(build_v5_scaffold(5)) == 5

    def test_key_format(self):
        result = build_v5_scaffold(3)
        assert result[0]["key"] == "graph1"
        assert result[2]["key"] == "graph3"

    def test_required_fields(self):
        for entry in build_v5_scaffold(5):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_question_is_uniform_natural_language(self):
        result = build_v5_scaffold(10)
        questions = {e["question"] for e in result}
        assert questions == {"What is the answer to this query?"}

    def test_answer_is_uniform_natural_language(self):
        result = build_v5_scaffold(10)
        answers = {e["answer"] for e in result}
        assert answers == {"The answer is currently unknown."}

    def test_no_duplicate_keys(self):
        result = build_v5_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_zero_keys(self):
        assert build_v5_scaffold(0) == []

    def test_start_index(self):
        result = build_v5_scaffold(2, start_index=3)
        assert result[0]["key"] == "graph3"
        assert result[1]["key"] == "graph4"

    def test_variant_builders_dispatch(self):
        assert VARIANT_BUILDERS[V5](5) == build_v5_scaffold(5)


# ---------------------------------------------------------------------------
# V6 scaffold — uniform short non-natural sentinel
# ---------------------------------------------------------------------------


class TestBuildV6Scaffold:
    """V6 builder produces uniform '<PLACEHOLDER>' across all slots."""

    def test_length(self):
        assert len(build_v6_scaffold(5)) == 5

    def test_required_fields(self):
        for entry in build_v6_scaffold(5):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_question_uniform_placeholder_token(self):
        result = build_v6_scaffold(10)
        assert {e["question"] for e in result} == {"<PLACEHOLDER>"}

    def test_answer_uniform_placeholder_token(self):
        result = build_v6_scaffold(10)
        assert {e["answer"] for e in result} == {"<PLACEHOLDER>"}

    def test_no_duplicate_keys(self):
        result = build_v6_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_start_index(self):
        result = build_v6_scaffold(2, start_index=7)
        assert result[0]["key"] == "graph7"

    def test_variant_builders_dispatch(self):
        assert VARIANT_BUILDERS[V6](5) == build_v6_scaffold(5)


# ---------------------------------------------------------------------------
# V7 scaffold — per-slot deterministic sha256 hex (was V5 in earlier naming)
# ---------------------------------------------------------------------------


class TestBuildV7Scaffold:
    """V7 builder produces deterministic random-hex Q/A per slot."""

    def test_length(self):
        assert len(build_v7_scaffold(5)) == 5

    def test_key_format(self):
        result = build_v7_scaffold(3)
        assert result[0]["key"] == "graph1"
        assert result[2]["key"] == "graph3"

    def test_required_fields(self):
        for entry in build_v7_scaffold(5):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_question_is_16_hex_chars(self):
        import re

        for entry in build_v7_scaffold(5):
            assert re.fullmatch(r"[0-9a-f]{16}", entry["question"]), (
                f"Expected 16-char hex, got {entry['question']!r}"
            )

    def test_answer_is_16_hex_chars(self):
        import re

        for entry in build_v7_scaffold(5):
            assert re.fullmatch(r"[0-9a-f]{16}", entry["answer"]), (
                f"Expected 16-char hex, got {entry['answer']!r}"
            )

    def test_deterministic_same_index_same_value(self):
        r1 = build_v7_scaffold(10)
        r2 = build_v7_scaffold(10)
        assert r1 == r2

    def test_deterministic_from_index_alone(self):
        """Slot N produces the same string regardless of start_index offset."""
        import hashlib

        n = 7
        expected_q = hashlib.sha256(f"V7-Q-{n}".encode()).hexdigest()[:16]
        expected_a = hashlib.sha256(f"V7-A-{n}".encode()).hexdigest()[:16]
        result = build_v7_scaffold(1, start_index=n)
        assert result[0]["question"] == expected_q
        assert result[0]["answer"] == expected_a

    def test_q_and_a_differ_per_slot(self):
        result = build_v7_scaffold(5)
        for entry in result:
            assert entry["question"] != entry["answer"], f"Q and A should differ: {entry}"

    def test_no_duplicate_q_values_across_slots(self):
        result = build_v7_scaffold(20)
        questions = [e["question"] for e in result]
        assert len(questions) == len(set(questions))

    def test_no_duplicate_a_values_across_slots(self):
        result = build_v7_scaffold(20)
        answers = [e["answer"] for e in result]
        assert len(answers) == len(set(answers))

    def test_no_duplicate_keys(self):
        result = build_v7_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_zero_keys(self):
        assert build_v7_scaffold(0) == []

    def test_start_index(self):
        result = build_v7_scaffold(2, start_index=3)
        assert result[0]["key"] == "graph3"
        assert result[1]["key"] == "graph4"

    def test_variant_builders_dispatch(self):
        assert VARIANT_BUILDERS[V7](5) == build_v7_scaffold(5)


# ---------------------------------------------------------------------------
# V8 scaffold — uniform long OOD hex (fallback variant)
# ---------------------------------------------------------------------------


class TestBuildV8Scaffold:
    """V8 builder produces a single shared hex string across all slots."""

    def test_length(self):
        assert len(build_v8_scaffold(5)) == 5

    def test_required_fields(self):
        for entry in build_v8_scaffold(5):
            assert set(entry.keys()) == {"key", "question", "answer"}

    def test_question_uniform_hex(self):
        result = build_v8_scaffold(10)
        assert {e["question"] for e in result} == {"a1b2c3d4e5f60718"}

    def test_answer_uniform_hex(self):
        result = build_v8_scaffold(10)
        assert {e["answer"] for e in result} == {"a1b2c3d4e5f60718"}

    def test_no_duplicate_keys(self):
        result = build_v8_scaffold(50)
        keys = [e["key"] for e in result]
        assert len(keys) == len(set(keys))

    def test_variant_builders_dispatch(self):
        assert VARIANT_BUILDERS[V8](5) == build_v8_scaffold(5)


# ---------------------------------------------------------------------------
# V3_extended scaffold (reuses V3 builder)
# ---------------------------------------------------------------------------


class TestV3ExtendedScaffold:
    """V3_extended maps to the same builder as V3 in VARIANT_BUILDERS."""

    def test_v3_extended_builder_is_v3_builder(self):
        """VARIANT_BUILDERS[V3_EXTENDED] and VARIANT_BUILDERS[V3] are the same function."""
        from experiments.utils.scaffold import build_v3_scaffold

        assert VARIANT_BUILDERS[V3_EXTENDED] is build_v3_scaffold

    def test_v3_extended_produces_identical_output_to_v3(self):
        """V3_extended and V3 builders produce the same scaffold for the same N."""
        assert VARIANT_BUILDERS[V3_EXTENDED](10) == VARIANT_BUILDERS[V3](10)

    def test_v3_extended_key_format(self):
        result = VARIANT_BUILDERS[V3_EXTENDED](3)
        assert [e["key"] for e in result] == ["graph1", "graph2", "graph3"]

    def test_v3_extended_uniform_sentinel(self):
        result = VARIANT_BUILDERS[V3_EXTENDED](5)
        for entry in result:
            assert entry["question"] == "pending"
            assert entry["answer"] == "pending"


# ---------------------------------------------------------------------------
# All-variants key structure parity (extended)
# ---------------------------------------------------------------------------


class TestAllVariantsKeyStructure:
    """All variants that include a key field use the graph{N} convention."""

    def test_variants_with_key_question_answer_structure(self):
        """V1/V2/V3/V3_extended/V5 all produce graph{i} keys with q+a fields."""
        for variant in (V1, V2, V3, V3_EXTENDED, V5):
            result = VARIANT_BUILDERS[variant](5)
            expected_keys = [f"graph{i}" for i in range(1, 6)]
            assert [e["key"] for e in result] == expected_keys, f"{variant}: key structure mismatch"
            for entry in result:
                assert "question" in entry, f"{variant}: missing question"
                assert "answer" in entry, f"{variant}: missing answer"

    def test_v4_empty_qa_structure(self):
        """V4 produces graph{i} keys with empty-string q+a fields."""
        result = VARIANT_BUILDERS[V4](5)
        expected_keys = [f"graph{i}" for i in range(1, 6)]
        assert [e["key"] for e in result] == expected_keys
        for entry in result:
            assert entry["question"] == ""
            assert entry["answer"] == ""
