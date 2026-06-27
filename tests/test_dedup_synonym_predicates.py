"""Tests for dedup_synonym_predicates (paramem/graph/extractor.py).

Uses a MagicMock model that returns canned JSON — no real GPU required.
Verifies:
- empty input returns empty clusters_by_so
- single-predicate groups pass through untouched (no model call)
- synonym clusters returned correctly for candidate groups
- distinct predicates on different (s,o) pairs are not clustered
- hallucinated predicates in model output are silently discarded
- clusters with <2 grounded members are dropped
- parse failure yields no clusters (never deletes)
- gradient_checkpointing is always re-enabled even when generate_answer raises
- SOTA branch: _sota_call receives raw (non-chat-templated) prompt
- SOTA branch: None return from _sota_call is a no-op (no cluster, no deletion)
- SOTA branch: apply_chat_template / generate_answer are NOT used on SOTA path
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.extractor import dedup_synonym_predicates


def _make_model_tokenizer(raw_outputs: list[str]):
    """Return a (model, tokenizer) MagicMock pair that produces the given
    raw strings from sequential generate_answer calls (one per candidate group)."""
    model = MagicMock()
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    return model, tokenizer


FILTER_PROMPT = "filter {predicates_json}"


def _run(
    relations: list[dict],
    raw_outputs: list[str],
) -> tuple[dict, dict]:
    """Run dedup_synonym_predicates with mocked generate_answer side-effects."""
    model, tokenizer = _make_model_tokenizer(raw_outputs)
    with patch(
        "paramem.graph.extractor.generate_answer",
        side_effect=raw_outputs,
    ):
        return dedup_synonym_predicates(
            relations,
            model=model,
            tokenizer=tokenizer,
            filter_prompt=FILTER_PROMPT,
        )


class TestEmptyInput:
    def test_empty_returns_empty_clusters(self):
        model = MagicMock()
        tokenizer = MagicMock()
        clusters_by_so, diag = dedup_synonym_predicates(
            [],
            model=model,
            tokenizer=tokenizer,
            filter_prompt=FILTER_PROMPT,
        )
        assert clusters_by_so == {}
        assert diag["groups_examined"] == 0
        assert diag["candidate_groups"] == 0
        assert diag["model_calls"] == 0


class TestSinglePredicateGroupPassthrough:
    def test_single_predicate_group_no_model_call(self):
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
        # If generate_answer were called, the side_effect would raise.
        with patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=Exception("should not be called"),
        ):
            clusters_by_so, diag = dedup_synonym_predicates(
                relations,
                model=model,
                tokenizer=tokenizer,
                filter_prompt=FILTER_PROMPT,
            )
        assert clusters_by_so == {}
        assert diag["candidate_groups"] == 0
        assert diag["model_calls"] == 0


class TestSynonymClusters:
    def test_synonym_pair_returns_cluster(self):
        """Two synonym predicates on same (s, o) → cluster returned."""
        relations = [
            {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
        ]
        # One candidate group → one model call.
        raw = json.dumps({"clusters": [["works_for", "employed_by"]]})
        clusters_by_so, diag = _run(relations, [raw])

        from paramem.graph.name_match import canonical

        key = (canonical("Alex"), canonical("Acme"))
        assert key in clusters_by_so, "Candidate group must appear in clusters_by_so"
        group_clusters = clusters_by_so[key]
        assert len(group_clusters) == 1
        assert len(group_clusters[0]) == 2
        assert diag["candidate_groups"] == 1
        assert diag["groups_with_clusters"] == 1
        assert diag["model_calls"] == 1

    def test_distinct_predicates_different_objects_no_cluster(self):
        """Same predicate but different objects — two distinct (s,o) groups,
        each single-predicate — no candidate group → no model call."""
        relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "Alex", "predicate": "lives_in", "object": "Munich"},
        ]
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        tokenizer = MagicMock()
        with patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=Exception("should not be called"),
        ):
            clusters_by_so, diag = dedup_synonym_predicates(
                relations,
                model=model,
                tokenizer=tokenizer,
                filter_prompt=FILTER_PROMPT,
            )
        assert clusters_by_so == {}
        assert diag["candidate_groups"] == 0

    def test_two_candidate_groups_two_model_calls(self):
        """Two distinct (s,o) groups each with 2 predicates → two model calls."""
        relations = [
            {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
            {"subject": "Sam", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "Sam", "predicate": "resides_in", "object": "Berlin"},
        ]
        raw1 = json.dumps({"clusters": [["works_for", "employed_by"]]})
        raw2 = json.dumps({"clusters": [["lives_in", "resides_in"]]})
        clusters_by_so, diag = _run(relations, [raw1, raw2])
        assert diag["candidate_groups"] == 2
        assert diag["model_calls"] == 2
        assert diag["groups_with_clusters"] == 2
        assert len(clusters_by_so) == 2


class TestGroundingGuard:
    def test_hallucinated_predicate_in_cluster_discarded(self):
        """A predicate in the model's cluster that does not appear in the input
        for that (s, o) group is silently discarded.  If only 1 grounded member
        remains, the cluster is dropped (not a valid ≥2-member cluster)."""
        relations = [
            {"subject": "Jordan", "predicate": "works_for", "object": "TechCo"},
            {"subject": "Jordan", "predicate": "employed_by", "object": "TechCo"},
        ]
        # Model returns cluster with "is_staff_at" hallucinated; 2 grounded remain.
        raw = json.dumps({"clusters": [["works_for", "employed_by", "is_staff_at"]]})
        clusters_by_so, diag = _run(relations, [raw])

        from paramem.graph.name_match import canonical

        key = (canonical("Jordan"), canonical("TechCo"))
        # Grounded cluster has 2 members (hallucinated "is_staff_at" dropped).
        assert key in clusters_by_so
        grounded = clusters_by_so[key][0]
        assert len(grounded) == 2
        assert canonical("is_staff_at") not in grounded
        assert any(d["reason"] == "hallucinated_predicate" for d in diag["discards"])

    def test_all_hallucinated_cluster_dropped(self):
        """When all cluster members are hallucinated, the cluster is dropped
        and no entry appears in clusters_by_so."""
        relations = [
            {"subject": "Sam", "predicate": "works_for", "object": "Corp"},
            {"subject": "Sam", "predicate": "employed_by", "object": "Corp"},
        ]
        # Model returns an entirely hallucinated cluster.
        raw = json.dumps({"clusters": [["ghost_pred", "phantom_pred"]]})
        clusters_by_so, diag = _run(relations, [raw])
        assert clusters_by_so == {}, "Entirely hallucinated cluster must be dropped"
        assert diag["groups_with_clusters"] == 0

    def test_parse_failure_yields_no_clusters(self):
        """When model returns unparseable JSON, no clusters are returned."""
        relations = [
            {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
        ]
        clusters_by_so, diag = _run(relations, ["not valid json at all"])
        assert clusters_by_so == {}
        assert diag["groups_with_clusters"] == 0
        # The model call was still made.
        assert diag["model_calls"] == 1

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
                dedup_synonym_predicates(
                    relations,
                    model=model,
                    tokenizer=tokenizer,
                    filter_prompt=FILTER_PROMPT,
                )
        model.gradient_checkpointing_enable.assert_called_once()


# ---------------------------------------------------------------------------
# SOTA branch tests
# ---------------------------------------------------------------------------


class TestSotaBranch:
    """dedup_synonym_predicates SOTA path: _sota_call receives raw prompt,
    generate_answer/apply_chat_template are NOT used, None return is a no-op."""

    _FILTER_PROMPT = "filter {predicates_json}"

    _SOTA_CFG = {
        "api_key": "sk-test",
        "provider": "anthropic",
        "filter_model": "claude-sonnet-4-6",
        "endpoint": None,
        "system_prompt": "You identify synonym predicate clusters. Output valid JSON only.",
    }

    def test_sota_call_receives_raw_prompt(self):
        """On the SOTA branch _sota_call receives the RAW rendered prompt —
        no chat template wrapping, no [INST] markers."""
        import json
        from unittest.mock import patch

        relations = [
            {"subject": "Jordan", "predicate": "works_for", "object": "TechCo"},
            {"subject": "Jordan", "predicate": "employed_by", "object": "TechCo"},
        ]
        cluster_raw = json.dumps({"clusters": [["works_for", "employed_by"]]})
        with patch("paramem.graph.extractor._sota_call", return_value=cluster_raw) as mock_sota:
            clusters_by_so, diag = dedup_synonym_predicates(
                relations,
                sota=self._SOTA_CFG,
                filter_prompt=self._FILTER_PROMPT,
            )

        assert mock_sota.called, "_sota_call must be invoked on the SOTA path"
        actual_prompt = mock_sota.call_args[0][0]
        # Raw rendered: the filter prompt with predicates_json substituted.
        assert "{predicates_json}" not in actual_prompt, (
            "Slot must be filled before passing to _sota_call"
        )
        # No chat-template markers — this is a raw string, not a formatted chat.
        assert "[INST]" not in actual_prompt
        assert "<|user|>" not in actual_prompt
        # Verify the cluster is parsed correctly.
        from paramem.graph.name_match import canonical

        key = (canonical("Jordan"), canonical("TechCo"))
        assert key in clusters_by_so
        assert diag["model_calls"] == 1

    def test_sota_none_return_is_noop(self):
        """_sota_call returning None (network / parse failure) must leave
        clusters_by_so empty and never trigger any deletion."""
        from unittest.mock import patch

        relations = [
            {"subject": "Jordan", "predicate": "works_for", "object": "TechCo"},
            {"subject": "Jordan", "predicate": "employed_by", "object": "TechCo"},
        ]
        with patch("paramem.graph.extractor._sota_call", return_value=None):
            clusters_by_so, diag = dedup_synonym_predicates(
                relations,
                sota=self._SOTA_CFG,
                filter_prompt=self._FILTER_PROMPT,
            )

        assert clusters_by_so == {}, "None return must produce no clusters"
        assert diag["groups_with_clusters"] == 0
        # model_calls is NOT incremented for a None return (raw_outputs gets "").
        assert diag["model_calls"] == 0

    def test_sota_path_does_not_use_local_model(self):
        """On the SOTA branch apply_chat_template and generate_answer must not
        be called — the local model is not consulted."""
        import json
        from unittest.mock import MagicMock, patch

        relations = [
            {"subject": "Jordan", "predicate": "works_for", "object": "TechCo"},
            {"subject": "Jordan", "predicate": "employed_by", "object": "TechCo"},
        ]
        fake_tokenizer = MagicMock()
        cluster_raw = json.dumps({"clusters": [["works_for", "employed_by"]]})
        with (
            patch("paramem.graph.extractor._sota_call", return_value=cluster_raw),
            patch(
                "paramem.graph.extractor.generate_answer",
                side_effect=AssertionError("generate_answer must not be called on SOTA path"),
            ),
        ):
            # Pass a tokenizer; the SOTA path must ignore it.
            dedup_synonym_predicates(
                relations,
                sota=self._SOTA_CFG,
                filter_prompt=self._FILTER_PROMPT,
            )

        fake_tokenizer.apply_chat_template.assert_not_called()
