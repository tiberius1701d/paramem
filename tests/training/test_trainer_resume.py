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


def _make_peft_model(adapter_name: str = "episodic") -> MagicMock:
    """Return a PeftModel stub that satisfies train_adapter's staging+promote contract.

    Both the production slot *adapter_name* and the staging slot ``in_training``
    are present with matching rank/target_modules so ``_ensure_staging_slot``
    is a no-op.  ``named_parameters`` yields one real tensor per
    ``(target_module, slot)`` pair so ``copy_adapter_weights`` finds parallel
    src/dst key sets at entry (production→staging) and at promote
    (staging→production).
    """
    import torch

    ac = _minimal_adapter_config()
    model = MagicMock()
    prod_cfg = MagicMock()
    prod_cfg.r = ac.rank
    prod_cfg.target_modules = set(ac.target_modules)
    staging_cfg = MagicMock()
    staging_cfg.r = ac.rank
    staging_cfg.target_modules = set(ac.target_modules)
    model.peft_config = {adapter_name: prod_cfg, "in_training": staging_cfg}

    named_params: list[tuple[str, "torch.Tensor"]] = []
    for module in sorted(ac.target_modules):
        for slot in (adapter_name, "in_training"):
            named_params.append((f"base_model.model.{module}.{slot}.weight", torch.zeros(1)))
    model.named_parameters.return_value = named_params
    model.parameters.return_value = [t for _, t in named_params]
    # Methods called inside train_adapter — let the mock absorb them.
    model.gradient_checkpointing_enable.return_value = None
    model.set_adapter.return_value = None
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


class TestTrainAdapterEncryptedResume:
    """Symmetry with EncryptCheckpointCallback: when Security is ON the
    on-disk ``checkpoint-N/`` is age-encrypted and HF Trainer's
    ``_load_from_checkpoint`` would crash on the age magic.  ``train_adapter``
    must materialise the checkpoint into a tempdir (decrypting en route via
    ``materialize_checkpoint_to_shm``) and hand HF the plaintext path,
    cleaning up the tempdir in ``finally``.  Mirrors the pattern already in
    ``BackgroundTrainer._train_adapter``.
    """

    def setup_method(self):
        _CapturingTrainer.captured_train_kwargs.clear()

    @pytest.fixture()
    def train_adapter_mocks(self, tmp_path):
        with (
            patch("paramem.training.trainer.TrainingArguments") as mock_args_cls,
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
        ):
            mock_args = MagicMock()
            mock_args_cls.return_value = mock_args
            yield tmp_path

    def test_security_off_skips_materialization(self, train_adapter_mocks, tmp_path):
        """When daily_identity_loadable returns False, the on-disk path is
        forwarded unchanged — no shm tempdir is created."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        ckpt = tmp_path / "adapter" / "checkpoint-40"

        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
            patch(
                "paramem.backup.checkpoint_shard.materialize_checkpoint_to_shm"
            ) as mock_materialize,
        ):
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

        mock_materialize.assert_not_called()
        forwarded = _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"]
        assert forwarded == str(ckpt)

    def test_security_on_materializes_to_shm_and_forwards_plaintext_path(
        self, train_adapter_mocks, tmp_path
    ):
        """When daily_identity_loadable returns True, train_adapter calls
        materialize_checkpoint_to_shm and hands HF Trainer the shm tempdir."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        ckpt = tmp_path / "adapter" / "checkpoint-40"
        shm_dir = tmp_path / "shm-mock"
        shm_dir.mkdir()

        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=True),
            patch(
                "paramem.backup.checkpoint_shard.materialize_checkpoint_to_shm",
                return_value=shm_dir,
            ) as mock_materialize,
            patch("paramem.training.trainer.shutil.rmtree") as mock_rmtree,
        ):
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

        mock_materialize.assert_called_once_with(Path(str(ckpt)))
        forwarded = _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"]
        assert forwarded == str(shm_dir)
        mock_rmtree.assert_called_once_with(shm_dir, ignore_errors=True)

    def test_security_on_no_resume_path_skips_materialization(self, train_adapter_mocks, tmp_path):
        """No resume_from_checkpoint → no shm work even with Security ON."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()

        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=True),
            patch(
                "paramem.backup.checkpoint_shard.materialize_checkpoint_to_shm"
            ) as mock_materialize,
        ):
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

        mock_materialize.assert_not_called()
        assert _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"] is None

    def test_shm_dir_cleaned_up_on_exception(self, train_adapter_mocks, tmp_path):
        """If trainer.train raises, the shm tempdir must still be removed."""
        tmp_path = train_adapter_mocks
        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        ckpt = tmp_path / "adapter" / "checkpoint-40"
        shm_dir = tmp_path / "shm-cleanup"
        shm_dir.mkdir()

        class _BoomTrainer(_CapturingTrainer):
            def train(self, resume_from_checkpoint=None):
                _CapturingTrainer.captured_train_kwargs.append(
                    {"resume_from_checkpoint": resume_from_checkpoint}
                )
                raise RuntimeError("boom")

        with (
            patch("paramem.training.trainer.Trainer", new=_BoomTrainer),
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=True),
            patch(
                "paramem.backup.checkpoint_shard.materialize_checkpoint_to_shm",
                return_value=shm_dir,
            ),
            patch("paramem.training.trainer.shutil.rmtree") as mock_rmtree,
            pytest.raises(RuntimeError, match="boom"),
        ):
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

        mock_rmtree.assert_called_once_with(shm_dir, ignore_errors=True)


class TestTrainAdapterSavePath:
    """Regression: save_pretrained gets output_dir.parent so PEFT's auto-append
    of the adapter name lands the weights at output_dir, not at
    output_dir/adapter_name/. Without this, infra_paths' rglob picks up
    unencrypted training-workspace safetensors and trips the startup
    mode-consistency check.
    """

    @pytest.fixture()
    def train_adapter_mocks(self, tmp_path):
        with (
            patch("paramem.training.trainer.TrainingArguments") as mock_args_cls,
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
        ):
            mock_args = MagicMock()
            mock_args_cls.return_value = mock_args
            yield tmp_path

    def test_trainer_does_not_save_canonical_adapter(self, train_adapter_mocks):
        """``train_adapter`` is responsible only for training.

        The canonical encrypted slot-dir save is the orchestrator's job
        (``ConsolidationLoop._save_adapters`` → ``atomic_save_adapter`` →
        ``_encrypt_adapter_safetensors``).  Writing here would duplicate
        the canonical save AND leave a plaintext
        ``adapter_model.safetensors`` inside ``data/ha/adapters/`` which
        trips the next boot's encryption mode-consistency check.
        """
        tmp_path = train_adapter_mocks
        adapter_name = "episodic_interim_20260430T1200"
        model = _make_peft_model(adapter_name=adapter_name)
        tokenizer = _make_tokenizer()
        out = tmp_path / "interim_20260430T1200" / adapter_name

        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=_make_dataset(),
            adapter_name=adapter_name,
            training_config=_minimal_training_config(),
            adapter_config=_minimal_adapter_config(),
            output_dir=out,
        )

        assert not model.save_pretrained.called, (
            "train_adapter must not write the canonical adapter — that's "
            "the orchestrator's job (atomic_save_adapter)"
        )
        assert not tokenizer.save_pretrained.called, (
            "train_adapter must not write the tokenizer — that's the orchestrator's job"
        )
