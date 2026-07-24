"""Unit tests for paramem.graph.empty_cause.

Pure vocabulary: no model, no tokenizer, no cloud, no config. Carved out
of ``test_relation_build.py`` alongside the ``CAUSE_*``/``cause_kind``
move out of ``paramem.graph.relation_build`` — the vocabulary now
describes flow state (``StageState.empty_cause``) shared by
``relation_build.recovery_gate``, the flow tail stages
(``paramem.graph.flows``), and the ``enrich`` stage
(``paramem.graph.stage_enrich``), so its own classification behaviour is
tested at its own module rather than borrowing ``relation_build``'s file.
"""

from __future__ import annotations

import pytest

from paramem.graph.empty_cause import (
    CAUSE_ANON_JUDGE,
    CAUSE_CLOUD_EMPTY,
    CAUSE_DEANON_JUDGE,
    CAUSE_DEANON_SUBSTITUTION,
    CAUSE_SCHEMA_VALIDATION,
    CAUSE_UNATTRIBUTED,
    EMPTY_CAUSE_KIND,
    cause_kind,
)


class TestCauseClassification:
    @pytest.mark.parametrize(
        "cause,kind",
        [
            (CAUSE_CLOUD_EMPTY, "judgment"),
            (CAUSE_ANON_JUDGE, "judgment"),
            (CAUSE_DEANON_JUDGE, "judgment"),
            (CAUSE_DEANON_SUBSTITUTION, "breakage"),
            (CAUSE_SCHEMA_VALIDATION, "breakage"),
        ],
    )
    def test_each_site_is_classified(self, cause, kind):
        assert cause_kind(cause) == kind

    def test_every_constant_is_in_the_table(self):
        assert set(EMPTY_CAUSE_KIND) == {
            CAUSE_CLOUD_EMPTY,
            CAUSE_ANON_JUDGE,
            CAUSE_DEANON_SUBSTITUTION,
            CAUSE_DEANON_JUDGE,
            CAUSE_SCHEMA_VALIDATION,
        }
        assert set(EMPTY_CAUSE_KIND.values()) == {"judgment", "breakage"}

    def test_unknown_and_none_are_unattributed(self):
        assert cause_kind(None) == CAUSE_UNATTRIBUTED
        assert cause_kind("something_else") == CAUSE_UNATTRIBUTED
