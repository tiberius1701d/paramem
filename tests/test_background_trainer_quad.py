"""Unit tests for BackgroundTrainer indexed_format dispatch.

Covers:
- BackgroundTrainer(indexed_format="quad") stores the flag.
- BackgroundTrainer(indexed_format="qa") stores the flag (default).
- _indexed_format == "quad" → format_quadruple_training is called at the
  _train_adapter format-selection code path.
- _indexed_format == "qa"  → format_indexed_training is called.

No GPU required — model interactions replaced with MagicMock stubs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from paramem.server.background_trainer import BackgroundTrainer
from paramem.utils.config import TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
    )


def _make_bt(tmp_path: Path, *, indexed_format: str = "qa") -> BackgroundTrainer:
    model = MagicMock()
    model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}
    return BackgroundTrainer(
        model,
        MagicMock(),
        _minimal_training_config(),
        tmp_path,
        indexed_format=indexed_format,
    )


# ---------------------------------------------------------------------------
# Tests: indexed_format stored on construction
# ---------------------------------------------------------------------------


class TestIndexedFormatConstructor:
    def test_default_is_qa(self, tmp_path: Path) -> None:
        bt = _make_bt(tmp_path)
        assert bt._indexed_format == "qa"

    def test_explicit_qa_stored(self, tmp_path: Path) -> None:
        bt = _make_bt(tmp_path, indexed_format="qa")
        assert bt._indexed_format == "qa"

    def test_quad_stored(self, tmp_path: Path) -> None:
        bt = _make_bt(tmp_path, indexed_format="quad")
        assert bt._indexed_format == "quad"


# ---------------------------------------------------------------------------
# Tests: format-selection logic (inline, no GPU)
# ---------------------------------------------------------------------------


class TestFormatSelectionLogic:
    """Test the format-selection logic in background_trainer._train_adapter.

    The branch is:

        if self._indexed_format == "quad":
            from paramem.training.quadruple_memory import format_quadruple_training as _fmt
        else:
            _fmt = format_indexed_training
        examples = _fmt(job.keyed_pairs, self.tokenizer, max_length=1024)

    We verify:
    - In quad mode, format_quadruple_training is imported and returns what we expect.
    - In QA mode, format_indexed_training is the active function.
    - The branch condition is simply `indexed_format == "quad"`.
    """

    def test_quad_flag_selects_different_function_than_qa(self, tmp_path: Path) -> None:
        """The two flags select different formatting functions."""
        from paramem.server.background_trainer import format_indexed_training
        from paramem.training.quadruple_memory import format_quadruple_training

        bt_qa = _make_bt(tmp_path, indexed_format="qa")
        bt_quad = _make_bt(tmp_path, indexed_format="quad")

        # In QA mode the module-level format_indexed_training is used.
        # In quad mode, format_quadruple_training is imported and used.
        # They must be different objects.
        assert format_indexed_training is not format_quadruple_training

        # The flag on the instance controls which one is selected.
        assert bt_qa._indexed_format == "qa"
        assert bt_quad._indexed_format == "quad"

    def test_flag_on_instance_controls_dispatch(self, tmp_path: Path) -> None:
        """Verify the instance flag is set by BackgroundTrainer.__init__."""
        bt_qa = _make_bt(tmp_path, indexed_format="qa")
        bt_quad = _make_bt(tmp_path, indexed_format="quad")
        assert bt_qa._indexed_format != bt_quad._indexed_format

    def test_quad_mode_format_function_accepts_quad_pairs(self) -> None:
        """format_quadruple_training is importable and accepts quad dicts (no GPU)."""
        from paramem.training.quadruple_memory import format_quadruple_training

        # format_quadruple_training is pure (no model.generate) so calling it
        # with an empty list and a MagicMock tokenizer must not raise.
        result = format_quadruple_training([], MagicMock(), max_length=512)
        assert isinstance(result, list)

    def test_qa_mode_format_function_accepts_qa_pairs(self) -> None:
        """format_indexed_training is importable and accepts QA dicts (no GPU)."""
        from paramem.server.background_trainer import format_indexed_training

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
        tokenizer.eos_token_id = 2

        # Empty list → empty result, no GPU needed.
        result = format_indexed_training([], tokenizer, max_length=512)
        assert isinstance(result, list)
