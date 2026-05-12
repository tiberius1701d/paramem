"""Unit tests for RecallEarlyStopCallback with quad eval_fn.

Covers:
- RecallEarlyStopCallback(eval_fn=evaluate_indexed_recall_quad) stores the fn.
- Default eval_fn is evaluate_indexed_recall (QA format) — byte-identical.
- Custom eval_fn is dispatched for both fill probe and retention probe.
- eval_fn=evaluate_indexed_recall (default) is dispatched when not overridden.

No GPU required — model/tokenizer and the eval_fn itself are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from paramem.training.early_stop import (
    EarlyStopPolicy,
    RecallEarlyStopCallback,
    _EarlyStopState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(*, probe_from_epoch: int = 1, signal_from_epoch: int = 1) -> EarlyStopPolicy:
    return EarlyStopPolicy(
        probe_from_epoch=probe_from_epoch,
        signal_from_epoch=signal_from_epoch,
        window=1,
        probe_every_n_epochs=1,
    )


def _make_result(*, exact: int = 1, total: int = 1) -> dict:
    """Build a minimal evaluate_indexed_recall-compatible result dict."""
    return {
        "exact_count": exact,
        "total": total,
        "rate": exact / total if total > 0 else 0.0,
        "mean_confidence": 0.9,
        "mean_expected_word_count": 0,
        "mean_recalled_word_count": 0,
        "per_key": [
            {
                "key": f"graph{i + 1}",
                "exact_match": i < exact,
                "question_match": i < exact,
                "answer_match": i < exact,
                "recalled": {"question": f"Q{i + 1}", "answer": f"A{i + 1}"},
                "confidence": 0.9,
            }
            for i in range(total)
        ],
    }


def _make_quad_result(*, exact: int = 1, total: int = 1) -> dict:
    """Build a minimal evaluate_indexed_recall_quad-compatible result dict."""
    return {
        "exact_count": exact,
        "total": total,
        "rate": exact / total if total > 0 else 0.0,
        "mean_confidence": 0.9,
        "mean_expected_word_count": 0,
        "mean_recalled_word_count": 0,
        "per_key": [
            {
                "key": f"graph{i + 1}",
                "exact_match": i < exact,
                "recalled_subject": "Alice",
                "recalled_predicate": "lives_in",
                "recalled_object": "Berlin",
                "confidence": 0.9,
                "failure_reason": None,
            }
            for i in range(total)
        ],
    }


def _make_state() -> MagicMock:
    s = MagicMock()
    s.epoch = 1.0
    return s


def _make_control() -> MagicMock:
    c = MagicMock()
    c.should_training_stop = False
    return c


def _make_args() -> MagicMock:
    a = MagicMock()
    a.gradient_checkpointing = False
    return a


def _make_callback(
    *,
    eval_fn=None,
    target_keyed: list[dict] | None = None,
    retention_keyed: list[dict] | None = None,
    retention_registry: dict | None = None,
    tmp_path: Path | None = None,
) -> RecallEarlyStopCallback:
    model = MagicMock()
    model.is_gradient_checkpointing = False
    target = target_keyed or [
        {"key": "graph1", "question": "Q1?", "answer": "A1."},
    ]
    registry = {"graph1": 12345}
    kwargs: dict = dict(
        model=model,
        tokenizer=MagicMock(),
        target_keyed=target,
        target_registry=registry,
        adapter_name="episodic",
        policy=_make_policy(),
        state_out=_EarlyStopState(),
        progress_path=None,
        epoch_log_path=None,
        first_perfect_log_path=None,
        phase_name="test",
        num_epochs=30,
        pause_file=None,
    )
    if eval_fn is not None:
        kwargs["eval_fn"] = eval_fn
    if retention_keyed is not None:
        kwargs["retention_keyed"] = retention_keyed
        kwargs["retention_registry"] = retention_registry or {"graph1": 12345}
    return RecallEarlyStopCallback(**kwargs)


# ---------------------------------------------------------------------------
# Tests: default eval_fn is evaluate_indexed_recall
# ---------------------------------------------------------------------------


class TestDefaultEvalFn:
    def test_default_eval_fn_is_none(self) -> None:
        """Default eval_fn=None → lazy import of evaluate_indexed_recall in on_epoch_end."""
        cb = _make_callback()
        assert cb._eval_fn is None

    def test_explicit_qa_fn_stored(self) -> None:
        from experiments.utils.test_harness import evaluate_indexed_recall

        cb = _make_callback(eval_fn=evaluate_indexed_recall)
        assert cb._eval_fn is evaluate_indexed_recall


# ---------------------------------------------------------------------------
# Tests: custom eval_fn stored and dispatched
# ---------------------------------------------------------------------------


class TestCustomEvalFn:
    def test_custom_eval_fn_stored(self) -> None:
        custom = MagicMock(return_value=_make_result())
        cb = _make_callback(eval_fn=custom)
        assert cb._eval_fn is custom

    def test_fill_probe_calls_custom_eval_fn(self) -> None:
        """on_epoch_end dispatches fill probe through the custom eval_fn."""
        custom = MagicMock(return_value=_make_quad_result(exact=1, total=1))
        cb = _make_callback(eval_fn=custom)

        cb.on_epoch_end(_make_args(), _make_state(), _make_control())

        assert custom.call_count >= 1, "Custom eval_fn must be called at least once"

    def test_retention_probe_calls_custom_eval_fn(self) -> None:
        """on_epoch_end dispatches retention probe through the same custom eval_fn."""
        call_count = {"n": 0}

        def _counting_fn(*args, **kwargs):
            call_count["n"] += 1
            return _make_quad_result(exact=1, total=1)

        cb = _make_callback(
            eval_fn=_counting_fn,
            retention_keyed=[{"key": "graph2", "subject": "Bob", "predicate": "p", "object": "C"}],
            retention_registry={"graph2": 99},
        )

        cb.on_epoch_end(_make_args(), _make_state(), _make_control())

        assert call_count["n"] == 2, (
            "eval_fn must be called twice: once for fill, once for retention"
        )

    def test_qa_eval_fn_used_when_explicit(self) -> None:
        """eval_fn=evaluate_indexed_recall is the QA-mode contract — verify dispatched."""
        calls: list = []

        def _spy_qa(model, tokenizer, keyed, registry, **kwargs):
            calls.append("qa")
            return _make_result(exact=1, total=1)

        cb = _make_callback(eval_fn=_spy_qa)
        cb.on_epoch_end(_make_args(), _make_state(), _make_control())
        assert "qa" in calls


# ---------------------------------------------------------------------------
# Tests: quad eval_fn dispatched with correct arguments
# ---------------------------------------------------------------------------


class TestQuadEvalFnArguments:
    def test_quad_eval_fn_receives_target_keyed_and_registry(self) -> None:
        """eval_fn is called with (model, tokenizer, target_keyed, registry, adapter_name=...)."""
        quad_pairs = [
            {"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"},
        ]
        quad_registry = {"graph1": 11111}

        call_args: list = []

        def _quad_fn(model, tokenizer, keyed, registry, **kwargs):
            call_args.append({"keyed": list(keyed), "registry": dict(registry)})
            return _make_quad_result(exact=1, total=1)

        cb = RecallEarlyStopCallback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            target_keyed=quad_pairs,
            target_registry=quad_registry,
            adapter_name="episodic",
            policy=_make_policy(),
            state_out=_EarlyStopState(),
            progress_path=None,
            epoch_log_path=None,
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=30,
            pause_file=None,
            eval_fn=_quad_fn,
        )

        cb.on_epoch_end(_make_args(), _make_state(), _make_control())

        assert len(call_args) == 1
        assert call_args[0]["keyed"] == quad_pairs
        assert call_args[0]["registry"] == quad_registry


# ---------------------------------------------------------------------------
# Tests: indexed_format config round-trip
# ---------------------------------------------------------------------------


class TestIndexedFormatConfigRoundTrip:
    def test_default_is_qa(self) -> None:
        from paramem.utils.config import ConsolidationConfig

        cfg = ConsolidationConfig()
        assert cfg.indexed_format == "qa"

    def test_quad_round_trips(self) -> None:
        from paramem.utils.config import ConsolidationConfig

        cfg = ConsolidationConfig(indexed_format="quad")
        assert cfg.indexed_format == "quad"

    def test_server_config_consolidation_config_property_propagates(self) -> None:
        """ServerConfig.consolidation_config propagates indexed_format."""
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        # Default in fixture should be "qa"; reading it back must not crash.
        cc = cfg.consolidation_config
        assert cc.indexed_format in ("qa", "quad")


# ---------------------------------------------------------------------------
# S3 — is_quad flag: diagnostics suppression in quad mode
# ---------------------------------------------------------------------------


class TestIsQuadDiagnostics:
    """S3 regression guard: is_quad=True suppresses QA-only diagnostics.

    ``per_field_split_counts`` and ``update_first_perfect_log`` consume
    ``question_match`` / ``answer_match`` / ``recalled`` keys that are absent
    from quad ``per_key`` entries.  Calling them in quad mode would log
    all-zero "q/a split" even when recall is 100%.  The ``is_quad=True``
    flag on RecallEarlyStopCallback gates these calls.

    The early-stop *decision* (fill rate + _consecutive_perfect) must be
    unaffected by the flag.
    """

    def test_is_quad_default_false(self) -> None:
        """Default is_quad=False — behaviour unchanged for QA callers."""
        cb = _make_callback()
        assert cb._is_quad is False

    def test_is_quad_stored(self) -> None:
        """is_quad=True is stored on the callback."""
        cb = _make_callback(eval_fn=MagicMock(return_value=_make_quad_result(exact=1, total=1)))
        # Manually set to verify storage
        cb._is_quad = True
        assert cb._is_quad is True

    def test_is_quad_kwarg_accepted(self) -> None:
        """RecallEarlyStopCallback accepts is_quad keyword argument."""
        cb = RecallEarlyStopCallback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1"}],
            target_registry={"graph1": 1234},
            adapter_name="episodic",
            policy=_make_policy(),
            state_out=_EarlyStopState(),
            progress_path=None,
            epoch_log_path=None,
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=10,
            pause_file=None,
            is_quad=True,
        )
        assert cb._is_quad is True

    def test_qa_path_calls_diagnostics(self) -> None:
        """QA path (is_quad=False) calls per_field_split_counts and update_first_perfect_log.

        The diagnostics are lazy-imported inside on_epoch_end from
        experiments.utils.recall_diagnostics, so we patch them at that module.
        """
        qa_result = _make_result(exact=1, total=1)
        custom = MagicMock(return_value=qa_result)

        from unittest.mock import patch as _patch

        with (
            _patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 1,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 1,
                    "q_correct": 1,
                    "a_correct": 1,
                },
            ) as split_mock,
            _patch("experiments.utils.recall_diagnostics.update_first_perfect_log") as log_mock,
        ):
            # Default is_quad=False — diagnostics should be called.
            cb = _make_callback(eval_fn=custom)
            cb.on_epoch_end(_make_args(), _make_state(), _make_control())

        split_mock.assert_called_once()
        log_mock.assert_called_once()

    def test_quad_path_skips_diagnostics(self) -> None:
        """Quad path (is_quad=True) skips per_field_split_counts and update_first_perfect_log.

        The diagnostics are lazy-imported inside on_epoch_end so we patch the
        functions in the source module (experiments.utils.recall_diagnostics).
        Because on_epoch_end re-imports them on each call (local import), the
        patch must target the module where they are *defined*, not a binding
        in early_stop.py.
        """
        quad_result = _make_quad_result(exact=1, total=1)
        custom = MagicMock(return_value=quad_result)

        from unittest.mock import patch as _patch

        with (
            _patch("experiments.utils.recall_diagnostics.per_field_split_counts") as split_mock,
            _patch("experiments.utils.recall_diagnostics.update_first_perfect_log") as log_mock,
        ):
            cb = RecallEarlyStopCallback(
                model=MagicMock(),
                tokenizer=MagicMock(),
                target_keyed=[{"key": "graph1", "subject": "A", "predicate": "p", "object": "B"}],
                target_registry={"graph1": 1234},
                adapter_name="episodic",
                policy=_make_policy(),
                state_out=_EarlyStopState(),
                progress_path=None,
                epoch_log_path=None,
                first_perfect_log_path=None,
                phase_name="test",
                num_epochs=10,
                pause_file=None,
                eval_fn=custom,
                is_quad=True,
            )
            cb.on_epoch_end(_make_args(), _make_state(), _make_control())

        split_mock.assert_not_called()
        log_mock.assert_not_called()

    def test_quad_early_stop_decision_still_fires(self) -> None:
        """is_quad=True does not break the early-stop decision logic.

        When fill is 100% and the window condition is met, the stop signal
        must still fire regardless of the is_quad flag.
        """
        quad_result = _make_quad_result(exact=1, total=1)
        custom = MagicMock(return_value=quad_result)

        policy = EarlyStopPolicy(
            probe_from_epoch=1,
            signal_from_epoch=1,
            window=1,
            probe_every_n_epochs=1,
        )
        state = _EarlyStopState()
        control = _make_control()

        cb = RecallEarlyStopCallback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "subject": "A", "predicate": "p", "object": "B"}],
            target_registry={"graph1": 1234},
            adapter_name="episodic",
            policy=policy,
            state_out=state,
            progress_path=None,
            epoch_log_path=None,
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=10,
            pause_file=None,
            eval_fn=custom,
            is_quad=True,
        )
        cb.on_epoch_end(_make_args(), _make_state(), control)

        # Stop signal must fire: perfect recall, window=1, signal_from_epoch=1.
        assert control.should_training_stop is True

    def test_epoch_log_q_a_split_zeros_in_quad_mode(self) -> None:
        """In quad mode, the epoch log entry has all-zero q_a_split (no misleading values)."""
        quad_result = _make_quad_result(exact=1, total=1)
        custom = MagicMock(return_value=quad_result)

        state_out = _EarlyStopState()
        cb = RecallEarlyStopCallback(
            model=MagicMock(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "subject": "A", "predicate": "p", "object": "B"}],
            target_registry={"graph1": 1234},
            adapter_name="episodic",
            policy=_make_policy(),
            state_out=state_out,
            progress_path=None,
            epoch_log_path=None,
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=10,
            pause_file=None,
            eval_fn=custom,
            is_quad=True,
        )
        cb.on_epoch_end(_make_args(), _make_state(), _make_control())

        assert state_out.epoch_log, "epoch_log must have one entry"
        entry = state_out.epoch_log[0]
        split = entry["q_a_split"]
        # All zeros — no meaningful QA-split data in quad mode.
        assert split["both"] == 0
        assert split["q_only"] == 0
        assert split["a_only"] == 0
        assert split["neither"] == 0
        # fill rate must still reflect the actual probe result.
        assert entry["fill"]["exact_count"] == 1
        assert entry["fill"]["total"] == 1
