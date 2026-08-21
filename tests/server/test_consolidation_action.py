"""Unit tests for ``paramem.server.consolidation_action`` — the dispatch's
pure decision layer (vocabulary, ``stages_event``, and the content gate).

Every function here is same-arguments-same-answer: no ``_state``, no buffer
mutation, no disk touch beyond the interim-slot listing the gate already
does.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.server.consolidation_action import ConsolidationAction, consolidation_content_gate


class TestStagesEventVocabulary:
    """``stages_event`` is ``True`` for exactly AUTO, FULL, INTERIM,
    RECONCILE — never CALIBRATE or CALIBRATE_PENDING."""

    def test_staging_actions_report_true(self) -> None:
        for action in (
            ConsolidationAction.AUTO,
            ConsolidationAction.FULL,
            ConsolidationAction.INTERIM,
            ConsolidationAction.RECONCILE,
        ):
            assert action.stages_event is True, f"{action} must stage an event"

    def test_calibrate_actions_report_false(self) -> None:
        for action in (ConsolidationAction.CALIBRATE, ConsolidationAction.CALIBRATE_PENDING):
            assert action.stages_event is False, f"{action} must not stage an event"

    def test_every_member_is_covered_by_exactly_one_of_the_two_partitions(self) -> None:
        staging = {a for a in ConsolidationAction if a.stages_event}
        non_staging = {a for a in ConsolidationAction if not a.stages_event}
        assert staging == {
            ConsolidationAction.AUTO,
            ConsolidationAction.FULL,
            ConsolidationAction.INTERIM,
            ConsolidationAction.RECONCILE,
        }
        assert non_staging == {
            ConsolidationAction.CALIBRATE,
            ConsolidationAction.CALIBRATE_PENDING,
        }
        assert staging | non_staging == set(ConsolidationAction)
        assert staging & non_staging == set()


class TestContentGateCalibrateActions:
    """the content gate never noops CALIBRATE; it noops
    CALIBRATE_PENDING exactly as it noops INTERIM."""

    def _config(self, tmp_path, *, max_interim_count: int = 7) -> MagicMock:
        cfg = MagicMock()
        cfg.consolidation.max_interim_count = max_interim_count
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def test_calibrate_never_noops_even_with_zero_pending_and_named(self, tmp_path) -> None:
        result = consolidation_content_gate(
            ConsolidationAction.CALIBRATE,
            self._config(tmp_path),
            pending_count=0,
            named_count=0,
            memory_store=None,
        )
        assert result is None

    def test_calibrate_pending_noops_on_zero_pending_exactly_like_interim(self, tmp_path) -> None:
        cfg = self._config(tmp_path)
        calibrate_pending_result = consolidation_content_gate(
            ConsolidationAction.CALIBRATE_PENDING,
            cfg,
            pending_count=0,
            named_count=0,
            memory_store=None,
        )
        interim_result = consolidation_content_gate(
            ConsolidationAction.INTERIM,
            cfg,
            pending_count=0,
            named_count=0,
            memory_store=None,
        )
        assert calibrate_pending_result == interim_result == "noop_no_pending"

    def test_calibrate_pending_noops_on_no_named_sessions_exactly_like_interim(
        self, tmp_path
    ) -> None:
        cfg = self._config(tmp_path)
        calibrate_pending_result = consolidation_content_gate(
            ConsolidationAction.CALIBRATE_PENDING,
            cfg,
            pending_count=3,
            named_count=0,
            memory_store=None,
        )
        interim_result = consolidation_content_gate(
            ConsolidationAction.INTERIM,
            cfg,
            pending_count=3,
            named_count=0,
            memory_store=None,
        )
        assert calibrate_pending_result == interim_result == "noop_no_named"

    def test_calibrate_pending_proceeds_with_at_least_one_named_session(self, tmp_path) -> None:
        result = consolidation_content_gate(
            ConsolidationAction.CALIBRATE_PENDING,
            self._config(tmp_path),
            pending_count=1,
            named_count=1,
            memory_store=None,
        )
        assert result is None


class TestContentGateStagingActions:
    """Control coverage: FULL / INTERIM / RECONCILE still behave exactly as
    documented, so the CALIBRATE_PENDING/INTERIM parity assertion above is
    meaningful rather than accidental."""

    def test_reconcile_noops_on_an_empty_store(self) -> None:
        from paramem.memory.store import MemoryStore

        cfg = MagicMock()
        result = consolidation_content_gate(
            ConsolidationAction.RECONCILE,
            cfg,
            pending_count=0,
            named_count=0,
            memory_store=MemoryStore(),
        )
        assert result == "noop_no_stored_keys"

    def test_reconcile_proceeds_when_store_is_unprovable(self) -> None:
        cfg = MagicMock()
        result = consolidation_content_gate(
            ConsolidationAction.RECONCILE,
            cfg,
            pending_count=0,
            named_count=0,
            memory_store=None,
        )
        assert result is None

    def test_full_at_zero_interim_count_falls_through_to_the_pending_check(self, tmp_path) -> None:
        cfg = MagicMock()
        cfg.consolidation.max_interim_count = 0
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
        result = consolidation_content_gate(
            ConsolidationAction.FULL,
            cfg,
            pending_count=0,
            named_count=0,
            memory_store=None,
        )
        assert result == "noop_no_pending"
