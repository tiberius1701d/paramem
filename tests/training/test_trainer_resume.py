"""Unit tests for train_adapter resume_from_checkpoint parameter.

Verifies that ``train_adapter`` threads ``resume_from_checkpoint`` through to
``trainer.train()`` without altering its default (None) behaviour.

No GPU required — the HF Trainer is mocked throughout so these tests run in
< 1 second each.

Related GPU-level integration: see
``experiments/test_resume_round_trip.py`` which exercises
``BackgroundTrainer._train_adapter`` (a private Trainer instance) end-to-end
on real hardware.  This module specifically covers the public
``paramem.training.trainer.train_adapter`` interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from paramem.training.trainer import train_adapter
from paramem.utils.config import AdapterConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_peft_model() -> MagicMock:
    """Return a minimal PeftModel stub accepted by train_adapter."""
    model = MagicMock()
    model.peft_config = {"episodic": MagicMock()}
    # gradient_checkpointing_enable is called inside train_adapter
    model.gradient_checkpointing_enable.return_value = None
    model.set_adapter.return_value = None
    # save_pretrained is called after training
    model.save_pretrained.return_value = None
    return model


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.save_pretrained.return_value = None
    return tok


def _minimal_training_config(**overrides) -> TrainingConfig:
    cfg = TrainingConfig(
        num_epochs=2,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=64,
        warmup_steps=0,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        seed=42,
        save_strategy="no",
        save_total_limit=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _minimal_adapter_config() -> AdapterConfig:
    return AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])


def _make_dataset() -> list[dict]:
    """Two-item dummy dataset with minimal tokenized structure."""
    return [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]


class _CapturingTrainer:
    """Fake HF Trainer that records the ``resume_from_checkpoint`` kwarg."""

    captured_train_kwargs: list[dict[str, Any]] = []

    def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
        self._args = args

    def train(self, resume_from_checkpoint=None):
        _CapturingTrainer.captured_train_kwargs.append(
            {"resume_from_checkpoint": resume_from_checkpoint}
        )
        result = MagicMock()
        result.metrics = {"train_loss": 0.1, "train_runtime": 0.01}
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainAdapterResumeParam:
    """Verify resume_from_checkpoint is threaded to trainer.train()."""

    def setup_method(self):
        _CapturingTrainer.captured_train_kwargs.clear()

    @pytest.fixture()
    def train_adapter_mocks(self, tmp_path):
        """Patch TrainingArguments and Trainer so no real HF objects are built."""
        with (
            patch("paramem.training.trainer.TrainingArguments") as mock_args_cls,
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
        ):
            mock_args = MagicMock()
            mock_args_cls.return_value = mock_args
            yield tmp_path

    def test_default_none_passes_none_to_trainer_train(self, train_adapter_mocks):
        """When resume_from_checkpoint is omitted, trainer.train(None) is called."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()

        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=_make_dataset(),
            adapter_name="episodic",
            training_config=_minimal_training_config(),
            adapter_config=_minimal_adapter_config(),
            output_dir=tmp_path / "adapter",
        )

        assert len(_CapturingTrainer.captured_train_kwargs) == 1
        assert _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"] is None

    def test_resume_path_str_forwarded(self, train_adapter_mocks, tmp_path):
        """When a str path is supplied, it is forwarded as a str to trainer.train."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        ckpt = "/fake/adapter/checkpoint-40"

        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=_make_dataset(),
            adapter_name="episodic",
            training_config=_minimal_training_config(),
            adapter_config=_minimal_adapter_config(),
            output_dir=tmp_path / "adapter",
            resume_from_checkpoint=ckpt,
        )

        assert len(_CapturingTrainer.captured_train_kwargs) == 1
        forwarded = _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"]
        assert forwarded == ckpt

    def test_resume_path_path_object_forwarded_as_str(self, train_adapter_mocks, tmp_path):
        """When a Path object is supplied, it is converted to str before forwarding."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        ckpt = Path("/fake/adapter/checkpoint-40")

        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=_make_dataset(),
            adapter_name="episodic",
            training_config=_minimal_training_config(),
            adapter_config=_minimal_adapter_config(),
            output_dir=tmp_path / "adapter",
            resume_from_checkpoint=ckpt,
        )

        assert len(_CapturingTrainer.captured_train_kwargs) == 1
        forwarded = _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"]
        assert forwarded == str(ckpt)
        assert isinstance(forwarded, str)

    def test_explicit_none_forwards_none(self, train_adapter_mocks, tmp_path):
        """Explicit resume_from_checkpoint=None must also forward None to trainer.train."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()

        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=_make_dataset(),
            adapter_name="episodic",
            training_config=_minimal_training_config(),
            adapter_config=_minimal_adapter_config(),
            output_dir=tmp_path / "adapter",
            resume_from_checkpoint=None,
        )

        assert len(_CapturingTrainer.captured_train_kwargs) == 1
        assert _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"] is None

    def test_train_adapter_returns_metrics(self, train_adapter_mocks, tmp_path):
        """train_adapter must still return the metrics dict from trainer.train()."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()

        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=_make_dataset(),
            adapter_name="episodic",
            training_config=_minimal_training_config(),
            adapter_config=_minimal_adapter_config(),
            output_dir=tmp_path / "adapter",
            resume_from_checkpoint="/fake/checkpoint-40",
        )

        assert isinstance(metrics, dict)
        assert "train_loss" in metrics
