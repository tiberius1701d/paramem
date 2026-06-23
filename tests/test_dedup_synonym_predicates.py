"""Tests for dedup_synonym_predicates_local (paramem/graph/extractor.py).

Uses a MagicMock model that returns canned JSON — no real GPU required.
Verifies:
- single-predicate groups pass through untouched
- synonym clusters collapse to the survivor
- hallucinated / ungrounded model output is ignored
- empty input returns empty output
- grounding guard: cross-group predicate in model output is discarded
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.extractor import dedup_synonym_predicates_local


def _make_model_tokenizer(filter_raw: str, merge_raw: str):
    """Return a (model, tokenizer) MagicMock pair that produces the given
    raw strings from two sequential generate_answer calls."""
    model = MagicMock()
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    return model, tokenizer, [filter_raw, merge_raw]


FILTER_PROMPT = "filter {facts_json}"
MERGE_PROMPT = "merge {clusters_json}"


def _run(
    relations: list[dict],
    filter_raw: str,
    merge_raw: str,
) -> tuple[list[dict], dict]:
    model, tokenizer, side_effects = _make_model_tokenizer(filter_raw, merge_raw)
    with patch(
        "paramem.graph.extractor.generate_answer",
        side_effect=side_effects,
    ):
        return dedup_synonym_predicates_local(
            relations,
            model,
            tokenizer,
            filter_prompt=FILTER_PROMPT,
            merge_prompt=MERGE_PROMPT,
        )


class TestEmptyInput:
    def test_empty_returns_empty(self):
        model = MagicMock()
        tokenizer = MagicMock()
        surviving, diag = dedup_synonym_predicates_local(
            [],
            model,
            tokenizer,
            filter_prompt=FILTER_PROMPT,
            merge_prompt=MERGE_PROMPT,
        )
        assert surviving == []
        assert diag["groups_examined"] == 0
        assert diag["predicates_retired"] == 0


class TestSinglePredicateGroupPassthrough:
    def test_single_predicate_group_survives_untouched(self):
        """A (subject, object) group with only one predicate passes through
        without any model call (candidate_keys is empty)."""
        relations = [
            {"subject": "Alex", "predicate": "likes", "object": "hiking"},
            {"subject": "Sam", "predicate": "works_at", "object": "Acme"},
        ]
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        tokenizer = MagicMock()
        # If generate_answer were called, the side_effect would be empty and raise.
        with patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=Exception("should not be called"),
        ):
            surviving, diag = dedup_synonym_predicates_local(
                relations,
                model,
                tokenizer,
                filter_prompt=FILTER_PROMPT,
                merge_prompt=MERGE_PROMPT,
            )
        assert len(surviving) == 2
        assert diag["candidate_groups"] == 0
        assert diag["predicates_retired"] == 0


class TestSynonymCollapseToSurvivor:
    def test_synonym_pair_collapses(self):
        """Two synonym predicates on same (s, o) → only the kept one survives."""
        relations = [
            {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
        ]
        filter_raw = json.dumps(
            {
                "groups": [
                    {
                        "subject": "Alex",
                        "object": "Acme",
                        "clusters": [["works_for", "employed_by"]],
                    }
                ]
            }
        )
        merge_raw = json.dumps(
            {
                "merges": [
                    {
                        "subject": "Alex",
                        "object": "Acme",
                        "keep": "works_for",
                        "drop": ["employed_by"],
                    }
                ]
            }
        )
        surviving, diag = _run(relations, filter_raw, merge_raw)
        preds = {r["predicate"] for r in surviving}
        assert preds == {"works_for"}
        assert diag["predicates_retired"] == 1
        assert diag["groups_collapsed"] == 1

    def test_distinct_predicate_not_retired(self):
        """A predicate with a different object is NOT retired even if the model
        mistakenly lumps it with a synonym cluster."""
        relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "Alex", "predicate": "lives_in", "object": "Munich"},
        ]
        # Both have the same predicate but DIFFERENT objects — two distinct groups,
        # one predicate each, so candidate_keys is empty.
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        tokenizer = MagicMock()
        with patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=Exception("should not be called"),
        ):
            surviving, diag = dedup_synonym_predicates_local(
                relations,
                model,
                tokenizer,
                filter_prompt=FILTER_PROMPT,
                merge_prompt=MERGE_PROMPT,
            )
        assert len(surviving) == 2
        assert diag["predicates_retired"] == 0


class TestGroundingGuard:
    def test_hallucinated_predicate_in_drop_is_ignored(self):
        """A predicate in merge.drop that does not appear in the input for
        that (s, o) group is silently discarded."""
        relations = [
            {"subject": "Jordan", "predicate": "works_for", "object": "TechCo"},
            {"subject": "Jordan", "predicate": "employed_by", "object": "TechCo"},
        ]
        filter_raw = json.dumps(
            {
                "groups": [
                    {
                        "subject": "Jordan",
                        "object": "TechCo",
                        "clusters": [["works_for", "employed_by"]],
                    }
                ]
            }
        )
        # Model hallucinates "is_staff_at" in drop — not in input.
        merge_raw = json.dumps(
            {
                "merges": [
                    {
                        "subject": "Jordan",
                        "object": "TechCo",
                        "keep": "works_for",
                        "drop": ["employed_by", "is_staff_at"],
                    }
                ]
            }
        )
        surviving, diag = _run(relations, filter_raw, merge_raw)
        preds = {r["predicate"] for r in surviving}
        # "employed_by" grounded and retired; "is_staff_at" hallucinated → discarded.
        assert preds == {"works_for"}
        assert diag["predicates_retired"] == 1
        assert any(d["reason"] == "hallucinated_drop" for d in diag["discards"])

    def test_hallucinated_keep_predicate_discards_entire_entry(self):
        """When the model's keep predicate is not in the input, the entire
        merge entry is discarded (grounding guard) — nothing is retired."""
        relations = [
            {"subject": "Sam", "predicate": "works_for", "object": "Corp"},
            {"subject": "Sam", "predicate": "employed_by", "object": "Corp"},
        ]
        filter_raw = json.dumps(
            {
                "groups": [
                    {
                        "subject": "Sam",
                        "object": "Corp",
                        "clusters": [["works_for", "employed_by"]],
                    }
                ]
            }
        )
        # Model invents "member_of" as the keep predicate — not in input.
        merge_raw = json.dumps(
            {
                "merges": [
                    {
                        "subject": "Sam",
                        "object": "Corp",
                        "keep": "member_of",
                        "drop": ["works_for", "employed_by"],
                    }
                ]
            }
        )
        surviving, diag = _run(relations, filter_raw, merge_raw)
        # Entire entry discarded — both original predicates survive.
        assert len(surviving) == 2
        assert diag["predicates_retired"] == 0
        assert any(d["reason"] == "hallucinated_keep" for d in diag["discards"])

    def test_ungrounded_group_in_filter_output_is_discarded(self):
        """A group in stage-1 output whose (subject, object) does not match
        any candidate group is discarded before stage-2."""
        relations = [
            {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
        ]
        filter_raw = json.dumps(
            {
                "groups": [
                    # Valid group.
                    {
                        "subject": "Alex",
                        "object": "Acme",
                        "clusters": [["works_for", "employed_by"]],
                    },
                    # Hallucinated group — (Ghost, Corp) not in input.
                    {
                        "subject": "Ghost",
                        "object": "Corp",
                        "clusters": [["invented_pred"]],
                    },
                ]
            }
        )
        merge_raw = json.dumps(
            {
                "merges": [
                    {
                        "subject": "Alex",
                        "object": "Acme",
                        "keep": "works_for",
                        "drop": ["employed_by"],
                    }
                ]
            }
        )
        surviving, diag = _run(relations, filter_raw, merge_raw)
        preds = {r["predicate"] for r in surviving}
        assert preds == {"works_for"}
        assert diag["predicates_retired"] == 1
        assert any(d["reason"] == "ungrounded_group" for d in diag["discards"])

    def test_parse_failure_leaves_all_relations_intact(self):
        """When model returns unparseable JSON, no relations are retired."""
        relations = [
            {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
        ]
        surviving, diag = _run(relations, "not valid json at all", "also not json")
        # Both survive because parsing failed.
        assert len(surviving) == 2
        assert diag["predicates_retired"] == 0

    def test_gradient_checkpointing_always_reenabled(self):
        """gradient_checkpointing_enable is called in the finally block even
        when generate_answer raises."""
        relations = [
            {"subject": "A", "predicate": "p1", "object": "B"},
            {"subject": "A", "predicate": "p2", "object": "B"},
        ]
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<fmt>")
        with patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=RuntimeError("GPU exploded"),
        ):
            with pytest.raises(RuntimeError, match="GPU exploded"):
                dedup_synonym_predicates_local(
                    relations,
                    model,
                    tokenizer,
                    filter_prompt=FILTER_PROMPT,
                    merge_prompt=MERGE_PROMPT,
                )
        model.gradient_checkpointing_enable.assert_called_once()
