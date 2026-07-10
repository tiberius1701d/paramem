"""Unit tests for ``ParamemTrainer._save`` adapter selection and ``create_scheduler``.

Guards the fold of ``_FixedDecayTrainer`` + the (formerly-missing) selective-save
override into a single ``ParamemTrainer`` class. No GPU required — instances are
built via ``object.__new__`` (bare instance, no HF ``__init__``) for the direct
``_save`` / ``create_scheduler`` unit tests, and driven through ``train_adapter``
with the HF ``Trainer`` mocked out for the ``save_adapter_name`` threading tests,
mirroring the patterns in ``tests/training/test_trainer_resume.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import torch
from peft import PeftModel

from paramem.training.trainer import ParamemTrainer, train_adapter
from paramem.utils.config import AdapterConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/training/test_trainer_resume.py)
# ---------------------------------------------------------------------------


def _minimal_training_config(**overrides) -> TrainingConfig:
    cfg = TrainingConfig(
        num_epochs=1,
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
    return [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.save_pretrained.return_value = None
    return tok


def _make_staging_model(adapter_name: str = "episodic") -> MagicMock:
    """PeftModel stub satisfying train_adapter's staging+promote contract.

    Both the production slot *adapter_name* and the staging slot ``in_training``
    are present with matching rank/target_modules so ``_ensure_staging_slot``
    is a no-op, and ``named_parameters`` yields one real tensor per
    ``(target_module, slot)`` pair so ``copy_adapter_weights`` finds parallel
    src/dst key sets.
    """
    ac = _minimal_adapter_config()
    model = MagicMock()
    prod_cfg = MagicMock(r=ac.rank, target_modules=set(ac.target_modules))
    staging_cfg = MagicMock(r=ac.rank, target_modules=set(ac.target_modules))
    model.peft_config = {adapter_name: prod_cfg, "in_training": staging_cfg}

    named_params: list[tuple[str, torch.Tensor]] = []
    for module in sorted(ac.target_modules):
        for slot in (adapter_name, "in_training"):
            named_params.append((f"base_model.model.{module}.{slot}.weight", torch.zeros(1)))
    model.named_parameters.return_value = named_params
    model.parameters.return_value = [t for _, t in named_params]
    model.gradient_checkpointing_enable.return_value = None
    model.set_adapter.return_value = None
    model.save_pretrained.return_value = None
    return model


def _make_compose_model(adapter_name: str, other_adapters: list[str]) -> MagicMock:
    """PeftModel stub for compose-training mode (``active_adapters`` supplied).

    ``train_adapter``'s compose branch derives trainable-ness from live tensor
    ``requires_grad`` flags (the real ``PeftModel.set_requires_grad`` call is
    mocked out here — a no-op recording call — so the tensors must already
    carry ``requires_grad=True`` to satisfy the "gradients are live" guard at
    ``trainer.py`` compose-mode entry).
    """
    ac = _minimal_adapter_config()
    model = MagicMock()
    cfg = MagicMock(r=ac.rank, target_modules=set(ac.target_modules))
    model.peft_config = {adapter_name: cfg, **{name: cfg for name in other_adapters}}

    named_params: list[tuple[str, torch.Tensor]] = []
    for module in sorted(ac.target_modules):
        t = torch.zeros(1)
        t.requires_grad_(True)
        named_params.append((f"base_model.model.{module}.{adapter_name}.weight", t))
    model.named_parameters.return_value = named_params
    model.parameters.return_value = [t for _, t in named_params]
    model.base_model.set_adapter.return_value = None
    model.set_requires_grad.return_value = None
    model.save_pretrained.return_value = None
    return model


class _CapturingTrainer:
    """Fake HF Trainer that records the kwargs it was constructed with."""

    captured_init_kwargs: list[dict[str, Any]] = []

    def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
        self._args = args
        _CapturingTrainer.captured_init_kwargs.append(kwargs)

    def train(self, resume_from_checkpoint=None):
        result = MagicMock()
        result.metrics = {"train_loss": 0.1, "train_runtime": 0.01}
        return result


# ---------------------------------------------------------------------------
# _save — direct unit tests (proves selected_adapters is the fix)
# ---------------------------------------------------------------------------


class TestSaveSelectsOnlyTrainedAdapter:
    def test_save_selects_only_staging_adapter(self, tmp_path):
        """``_save`` must call ``model.save_pretrained`` exactly once with
        ``selected_adapters == ["in_training"]`` and ``state_dict`` threaded
        through unchanged.

        This is the direct regression proof: dropping ``selected_adapters``
        from the call (reverting to stock ``Trainer._save`` behaviour) would
        make PEFT serialize every attached adapter instead of just the one
        this training event mutated, and this assertion would fail.
        """
        trainer = object.__new__(ParamemTrainer)
        trainer._save_adapter_name = "in_training"
        trainer.model = MagicMock(spec=PeftModel)
        trainer.args = MagicMock(output_dir=str(tmp_path))
        trainer.processing_class = None
        trainer.data_collator = None

        with patch("paramem.training.trainer.torch.save"):
            trainer._save(str(tmp_path), state_dict={"foo": "bar"})

        trainer.model.save_pretrained.assert_called_once()
        call_args, call_kwargs = trainer.model.save_pretrained.call_args
        assert call_args[0] == str(tmp_path)
        assert call_kwargs["selected_adapters"] == ["in_training"]
        assert call_kwargs["state_dict"] == {"foo": "bar"}

    def test_save_uses_args_output_dir_when_output_dir_omitted(self, tmp_path):
        """When ``output_dir`` is omitted, ``self.args.output_dir`` is used —
        matches stock ``Trainer._save`` behaviour."""
        trainer = object.__new__(ParamemTrainer)
        trainer._save_adapter_name = "episodic"
        trainer.model = MagicMock(spec=PeftModel)
        trainer.args = MagicMock(output_dir=str(tmp_path))
        trainer.processing_class = None
        trainer.data_collator = None

        with patch("paramem.training.trainer.torch.save"):
            trainer._save(None, state_dict=None)

        call_args, call_kwargs = trainer.model.save_pretrained.call_args
        assert call_args[0] == str(tmp_path)
        assert call_kwargs["selected_adapters"] == ["episodic"]
        assert call_kwargs["state_dict"] is None

    def test_save_saves_tokenizer_and_training_args(self, tmp_path):
        """Parity check vs stock ``_save``: tokenizer save and
        ``torch.save(self.args, .../training_args.bin)`` still occur."""
        trainer = object.__new__(ParamemTrainer)
        trainer._save_adapter_name = "in_training"
        trainer.model = MagicMock(spec=PeftModel)
        trainer.args = MagicMock(output_dir=str(tmp_path))
        trainer.processing_class = None
        tokenizer = MagicMock()
        trainer.data_collator = MagicMock(tokenizer=tokenizer)

        with patch("paramem.training.trainer.torch.save") as mock_torch_save:
            trainer._save(str(tmp_path), state_dict=None)

        tokenizer.save_pretrained.assert_called_once_with(str(tmp_path))
        mock_torch_save.assert_called_once()
        saved_args, saved_path = mock_torch_save.call_args[0]
        assert saved_args is trainer.args
        assert saved_path.endswith("training_args.bin")

    def test_save_delegates_to_super_for_non_peft_model(self, tmp_path):
        """Non-``PeftModel`` instances fall through to stock ``Trainer._save``
        unchanged — no ``ParamemTrainer`` instantiation in this codebase hits
        this branch, but the fallback preserves stock behaviour."""
        trainer = object.__new__(ParamemTrainer)
        trainer._save_adapter_name = "in_training"
        trainer.model = MagicMock()  # not a PeftModel

        with patch("transformers.Trainer._save") as mock_super_save:
            trainer._save(str(tmp_path), state_dict=None)

        mock_super_save.assert_called_once_with(str(tmp_path), None)
        trainer.model.save_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# save_adapter_name threading through train_adapter (guards §3 of the plan)
# ---------------------------------------------------------------------------


class TestSaveTargetSelection:
    def setup_method(self):
        _CapturingTrainer.captured_init_kwargs.clear()

    def test_save_target_is_staging_when_use_staging(self, tmp_path):
        """Default (staging+promote) path trains ``in_training`` — the
        checkpoint save target must match."""
        model = _make_staging_model()
        tokenizer = _make_tokenizer()

        with (
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch("paramem.training.trainer.ParamemTrainer", new=_CapturingTrainer),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=_make_dataset(),
                adapter_name="episodic",
                training_config=_minimal_training_config(),
                adapter_config=_minimal_adapter_config(),
                output_dir=tmp_path,
            )

        assert _CapturingTrainer.captured_init_kwargs[0]["save_adapter_name"] == "in_training"

    def test_save_target_is_adapter_name_in_compose_mode(self, tmp_path):
        """Compose mode (``active_adapters`` supplied) trains *adapter_name*
        in place — ``in_training`` is never created, so selecting it would
        raise in PEFT. The save target must be *adapter_name*."""
        model = _make_compose_model(adapter_name="episodic", other_adapters=["semantic"])
        tokenizer = _make_tokenizer()

        with (
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch("paramem.training.trainer.ParamemTrainer", new=_CapturingTrainer),
        ):
            train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=_make_dataset(),
                adapter_name="episodic",
                training_config=_minimal_training_config(),
                adapter_config=_minimal_adapter_config(),
                output_dir=tmp_path,
                active_adapters=["episodic", "semantic"],
            )

        assert _CapturingTrainer.captured_init_kwargs[0]["save_adapter_name"] == "episodic"

    def test_lr_decay_steps_threaded_through_unconditionally(self, tmp_path):
        """``lr_decay_steps`` is always passed to ``ParamemTrainer`` (may be
        ``None``) — the old ``trainer_cls`` / ``trainer_kwargs`` conditional
        selection is gone."""
        model = _make_staging_model()
        tokenizer = _make_tokenizer()

        with (
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch("paramem.training.trainer.ParamemTrainer", new=_CapturingTrainer),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=_make_dataset(),
                adapter_name="episodic",
                training_config=_minimal_training_config(lr_decay_steps=250),
                adapter_config=_minimal_adapter_config(),
                output_dir=tmp_path,
            )

        assert _CapturingTrainer.captured_init_kwargs[0]["lr_decay_steps"] == 250


# ---------------------------------------------------------------------------
# create_scheduler — fixed-decay behaviour survived the fold
# ---------------------------------------------------------------------------


class TestCreateSchedulerNoop:
    def test_create_scheduler_noop_when_lr_decay_none(self):
        """``lr_decay_steps=None`` (production default) must not alter
        ``num_training_steps`` — a verbatim delegation to
        ``super().create_scheduler(...)``, proving unconditional
        ``ParamemTrainer`` instantiation is a no-op in production config."""
        trainer = object.__new__(ParamemTrainer)
        trainer._lr_decay_steps = None

        with patch("transformers.Trainer.create_scheduler") as mock_super:
            trainer.create_scheduler(num_training_steps=123, optimizer="opt")

        mock_super.assert_called_once_with(123, "opt")

    def test_create_scheduler_substitutes_when_lr_decay_set(self):
        """``lr_decay_steps`` set substitutes ``num_training_steps`` — sanity
        check that the fixed-decay behaviour survived the fold from
        ``_FixedDecayTrainer`` into ``ParamemTrainer``."""
        trainer = object.__new__(ParamemTrainer)
        trainer._lr_decay_steps = 500

        with patch("transformers.Trainer.create_scheduler") as mock_super:
            trainer.create_scheduler(num_training_steps=123, optimizer="opt")

        mock_super.assert_called_once_with(500, "opt")
