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

import json
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
        """Patch TrainingArguments and Trainer so no real HF objects are built.

        Also patches ``_ensure_staging_slot`` to a no-op: the fixture
        pre-populates ``peft_config`` with the in_training slot for tensor
        access, which would otherwise trip the AD-20 lifecycle invariant guard.
        """
        with (
            patch("paramem.training.trainer.TrainingArguments") as mock_args_cls,
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
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
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
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
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
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
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
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


# ---------------------------------------------------------------------------
# Content-stable fingerprint tests
# ---------------------------------------------------------------------------


class TestFingerprintDatasetContentStable:
    """Verify ``_fingerprint_dataset`` produces content-based, address-free digests.

    All tests are no-GPU.  ``_IndexedDataset`` is obtained via
    ``ConsolidationLoop._indexed_dataset`` (a staticmethod) so no
    model/tokenizer is needed.  Security is OFF (no daily identity) so
    plaintext JSON is written/read for the resume-match test.
    """

    def _make_items(self, label_value: int = 2) -> list[dict]:
        """Two pre-tokenized examples with 1-D torch tensors as values."""
        import torch

        return [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, label_value, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "labels": torch.tensor([-100, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
        ]

    def _indexed_dataset(self, items):
        from paramem.training.consolidation import ConsolidationLoop

        return ConsolidationLoop._indexed_dataset(items)

    def test_address_independence(self):
        """Two separately-constructed datasets with identical content hash equal.

        This is the core regression: the old ``str(object)`` code embedded the
        heap address, so two instances always differed.
        """
        from paramem.training.trainer import _fingerprint_dataset

        items_a = self._make_items()
        items_b = self._make_items()
        ds_a = self._indexed_dataset(items_a)
        ds_b = self._indexed_dataset(items_b)

        # Confirm the two objects are distinct (different heap addresses)
        assert ds_a is not ds_b

        assert _fingerprint_dataset(ds_a) == _fingerprint_dataset(ds_b)

    def test_content_sensitivity(self):
        """Changing one token value produces a different fingerprint."""
        from paramem.training.trainer import _fingerprint_dataset

        ds_a = self._indexed_dataset(self._make_items(label_value=2))
        ds_b = self._indexed_dataset(self._make_items(label_value=99))

        assert _fingerprint_dataset(ds_a) != _fingerprint_dataset(ds_b)

    def test_resolve_resume_checkpoint_matches_on_identical_content(self, tmp_path):
        """Writing a staging_resume.json with a content fingerprint, then
        resolving with a freshly-constructed identical-content dataset
        (simulating a process restart) returns the checkpoint — not None.

        Security is OFF (no daily identity) so the file is plaintext JSON.
        ``_resolve_resume_checkpoint`` reads via ``read_maybe_encrypted`` which
        transparently handles plaintext.
        """
        import json

        from paramem.training.trainer import _fingerprint_dataset, _resolve_resume_checkpoint

        # Build original dataset and fingerprint it
        ds_original = self._indexed_dataset(self._make_items())
        fp_dataset = _fingerprint_dataset(ds_original)
        fp_config = "aabbccdd" * 8  # stable stand-in for config fingerprint

        # Create a fake checkpoint directory that _resolve_resume_checkpoint will verify
        ckpt_dir = tmp_path / "bg_checkpoint_epoch"
        ckpt_dir.mkdir()

        # Write staging_resume.json as plaintext (Security OFF — no daily identity)
        resume_path = tmp_path / "staging_resume.json"
        state = {
            "dataset_fingerprint": fp_dataset,
            "training_config_fingerprint": fp_config,
            "disk_checkpoint_path": str(ckpt_dir),
        }
        resume_path.write_text(json.dumps(state))

        # Simulate restart: fresh identical-content dataset
        ds_fresh = self._indexed_dataset(self._make_items())

        fingerprints = {
            "dataset": _fingerprint_dataset(ds_fresh),
            "config": fp_config,
        }

        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
        ):
            result = _resolve_resume_checkpoint(resume_path, fingerprints)

        assert result == str(ckpt_dir), (
            f"Expected checkpoint path {ckpt_dir!s}, got {result!r}. "
            "Fingerprint mismatch — content-hash is not cross-restart stable."
        )

        # Negative: mutated content must NOT match
        ds_mutated = self._indexed_dataset(self._make_items(label_value=99))
        fingerprints_mutated = {
            "dataset": _fingerprint_dataset(ds_mutated),
            "config": fp_config,
        }
        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
        ):
            result_mutated = _resolve_resume_checkpoint(resume_path, fingerprints_mutated)

        assert result_mutated is None

    def test_empty_dataset_stable_fingerprint(self):
        """An empty dataset produces a stable non-crashing fingerprint."""
        from paramem.training.trainer import _fingerprint_dataset

        ds_empty = self._indexed_dataset([])
        fp1 = _fingerprint_dataset(ds_empty)
        fp2 = _fingerprint_dataset(ds_empty)

        assert isinstance(fp1, str) and len(fp1) == 64  # 32 bytes → 64 hex chars
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# Slice 2a — on_save records output_dir/checkpoint-* in default epoch-save mode
# ---------------------------------------------------------------------------


class TestStagingResumeCallbackOnSave:
    """Verify ``_StagingResumeCallback.on_save`` records the correct checkpoint path.

    Three scenarios:
    1. Default epoch-save mode (``save_steps_ram==0``): HF Trainer writes
       ``checkpoint-N`` directly under ``output_dir``; ``on_save`` must record
       that as ``disk_checkpoint_path`` and ``_resolve_resume_checkpoint`` must
       return it on a fresh-process simulation with matching fingerprints.
    2. When ``bg_checkpoint_epoch/`` exists (RAM copy-back mode), that dir is
       preferred over ``output_dir/checkpoint-*`` (existing behaviour preserved).
    3. A dir recorded by ``on_save`` that was subsequently deleted (cleaned) must
       cause ``_resolve_resume_checkpoint`` to return ``None``.

    All tests are no-GPU; no real HF Trainer is invoked.
    """

    def _write_resume(self, path: Path, state: dict) -> None:
        """Write plaintext staging_resume.json (Security OFF posture)."""
        path.write_bytes(json.dumps(state, indent=2).encode())

    def _read_resume(self, path: Path) -> dict:
        return json.loads(path.read_bytes())

    def _base_state(self, fp_dataset: str, fp_config: str) -> dict:
        return {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_dataset,
            "training_config_fingerprint": fp_config,
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": "",
            "checkpoint_path": "",
            "started_at": "2026-06-21T00:00:00+00:00",
            "updated_at": "2026-06-21T00:00:00+00:00",
        }

    def _invoke_on_save(self, cb):
        """Fire the on_save callback with encryption patched to plaintext."""
        write_plain = lambda p, d: Path(p).write_bytes(d)  # noqa: E731
        with patch("paramem.backup.encryption.write_infra_bytes", side_effect=write_plain):
            cb.on_save(args=MagicMock(), state=MagicMock(), control=MagicMock())

    def _read_resume_plain(self, scratch_path):
        """Read staging_resume.json with encryption patched to plaintext."""
        from paramem.training.trainer import _read_staging_resume

        read_plain = lambda p: Path(p).read_bytes()  # noqa: E731
        with patch("paramem.backup.encryption.read_maybe_encrypted", side_effect=read_plain):
            return _read_staging_resume(scratch_path)

    def test_on_save_records_output_dir_checkpoint_when_no_epoch_mirror(self, tmp_path):
        """Default epoch-save mode: on_save sets disk_checkpoint_path to output_dir/checkpoint-N.

        Simulates the consolidation fold's save mode where save_steps_ram==0 and
        HF Trainer writes checkpoint-N directly under output_dir (no bg_checkpoint_epoch/).
        """
        from paramem.training.trainer import _StagingResumeCallback

        output_dir = tmp_path / "consolidation_refresh" / "episodic"
        output_dir.mkdir(parents=True)

        # HF Trainer writes checkpoint-5 directly under output_dir.
        ckpt_dir = output_dir / "checkpoint-5"
        ckpt_dir.mkdir()

        fp_dataset = "aabbccdd" * 8
        fp_config = "11223344" * 8
        scratch_path = output_dir / "staging_resume.json"
        base_state = self._base_state(fp_dataset, fp_config)
        self._write_resume(scratch_path, base_state)

        cb = _StagingResumeCallback(
            scratch_path=scratch_path,
            ram_dir=None,
            output_dir=output_dir,
            base_state=base_state,
        )
        self._invoke_on_save(cb)

        state = self._read_resume_plain(scratch_path)
        assert state is not None
        assert state["disk_checkpoint_path"].endswith("checkpoint-5"), (
            f"Expected disk_checkpoint_path to end with 'checkpoint-5', "
            f"got {state['disk_checkpoint_path']!r}"
        )

    def test_on_save_resolves_output_dir_checkpoint_on_restart(self, tmp_path):
        """Simulates a fresh-process restart after crash in default epoch-save mode.

        Writes a staging_resume.json with disk_checkpoint_path set to
        output_dir/checkpoint-5 (as on_save would record), then verifies
        _resolve_resume_checkpoint returns that path on a restart with
        matching fingerprints.
        """
        import torch

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.trainer import _fingerprint_dataset, _resolve_resume_checkpoint

        output_dir = tmp_path / "consolidation_refresh" / "episodic"
        output_dir.mkdir(parents=True)

        items_tensored = [
            {
                k: torch.tensor(v)
                for k, v in {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}.items()
            }
        ]
        ds = ConsolidationLoop._indexed_dataset(items_tensored)
        fp_dataset = _fingerprint_dataset(ds)
        fp_config = "aabbccdd" * 8

        # Create the checkpoint dir (as HF Trainer would write it).
        ckpt_dir = output_dir / "checkpoint-5"
        ckpt_dir.mkdir()

        scratch_path = output_dir / "staging_resume.json"
        state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_dataset,
            "training_config_fingerprint": fp_config,
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": str(ckpt_dir),
            "checkpoint_path": "",
            "started_at": "2026-06-21T00:00:00+00:00",
            "updated_at": "2026-06-21T00:00:00+00:00",
        }
        scratch_path.write_bytes(json.dumps(state, indent=2).encode())

        # Simulate restart: fresh identical-content dataset.
        ds_fresh = ConsolidationLoop._indexed_dataset(items_tensored)
        fingerprints = {
            "dataset": _fingerprint_dataset(ds_fresh),
            "config": fp_config,
        }

        read_plain = lambda p: Path(p).read_bytes()  # noqa: E731
        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
            patch("paramem.backup.encryption.read_maybe_encrypted", side_effect=read_plain),
        ):
            result = _resolve_resume_checkpoint(scratch_path, fingerprints)

        assert result == str(ckpt_dir), (
            f"Expected {ckpt_dir!s}, got {result!r}. "
            "on_save fix must make _resolve_resume_checkpoint find output_dir/checkpoint-N."
        )

    def test_on_save_none_when_checkpoint_dir_cleaned(self, tmp_path):
        """_resolve_resume_checkpoint returns None when the recorded checkpoint was deleted.

        The is_dir() gate in _resolve_resume_checkpoint must reject a path whose
        directory no longer exists (cleaned by _clean_scratch on a prior success).
        """
        from paramem.training.trainer import _resolve_resume_checkpoint

        output_dir = tmp_path / "consolidation_refresh" / "episodic"
        output_dir.mkdir(parents=True)

        fp_dataset = "aabbccdd" * 8
        fp_config = "11223344" * 8

        scratch_path = output_dir / "staging_resume.json"
        # Record a checkpoint path pointing to a dir that does NOT exist.
        state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_dataset,
            "training_config_fingerprint": fp_config,
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": str(output_dir / "checkpoint-5"),  # dir absent
            "checkpoint_path": "",
            "started_at": "2026-06-21T00:00:00+00:00",
            "updated_at": "2026-06-21T00:00:00+00:00",
        }
        scratch_path.write_bytes(json.dumps(state, indent=2).encode())

        fingerprints = {"dataset": fp_dataset, "config": fp_config}

        read_plain = lambda p: Path(p).read_bytes()  # noqa: E731
        with (
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
            patch("paramem.backup.encryption.read_maybe_encrypted", side_effect=read_plain),
        ):
            result = _resolve_resume_checkpoint(scratch_path, fingerprints)

        assert result is None, f"Expected None when checkpoint dir was cleaned; got {result!r}"

    def test_on_save_epoch_mirror_takes_precedence_over_output_dir_checkpoint(self, tmp_path):
        """When bg_checkpoint_epoch/ exists, it is preferred over output_dir/checkpoint-*.

        This is the existing RAM copy-back mode behaviour; the new else-branch
        must not activate when the epoch mirror dir is present.
        """
        from paramem.training.trainer import _StagingResumeCallback

        output_dir = tmp_path / "adapter"
        output_dir.mkdir(parents=True)

        # Both bg_checkpoint_epoch/ and a plain checkpoint-N exist.
        epoch_mirror = output_dir / "bg_checkpoint_epoch"
        epoch_mirror.mkdir()
        (epoch_mirror / "checkpoint-10").mkdir()
        (output_dir / "checkpoint-5").mkdir()

        fp_dataset = "aabbccdd" * 8
        fp_config = "11223344" * 8
        scratch_path = output_dir / "staging_resume.json"
        base_state = self._base_state(fp_dataset, fp_config)
        self._write_resume(scratch_path, base_state)

        cb = _StagingResumeCallback(
            scratch_path=scratch_path,
            ram_dir=None,
            output_dir=output_dir,
            base_state=base_state,
        )
        self._invoke_on_save(cb)

        state = self._read_resume_plain(scratch_path)
        assert state is not None
        recorded = state["disk_checkpoint_path"]
        assert "bg_checkpoint_epoch" in recorded, (
            f"Expected bg_checkpoint_epoch to be preferred; got {recorded!r}"
        )
        assert "checkpoint-10" in recorded, (
            f"Expected checkpoint-10 from epoch mirror; got {recorded!r}"
        )


# ---------------------------------------------------------------------------
# Slice 2a (fresh-start purge) — stale checkpoint-N purged on fingerprint mismatch
# ---------------------------------------------------------------------------


class TestFreshStartStaleCheckpointPurge:
    """Verify that ``train_adapter`` purges stale ``checkpoint-*`` dirs on the
    fresh-start branch (fingerprint mismatch or first run) and does NOT purge
    them on the resume branch (matching fingerprints + valid checkpoint).

    Both tests are no-GPU; the HF Trainer is mocked throughout.

    The crash scenario: a fold crashes mid-tier leaving an age-encrypted
    ``checkpoint-N/README.md``.  The next cycle has DIFFERENT content (new
    sessions) so fingerprints mismatch → fresh-start branch.  PEFT's
    ``save_pretrained`` calls ``ModelCard.load(checkpoint-N/README.md)`` which
    opens the file as UTF-8 and crashes on the age magic bytes.  The fix calls
    ``_clean_scratch(output_dir, ram_dir=None)`` at the TOP of the fresh-start
    block, before ``_write_staging_resume``.
    """

    # AGE file magic: first 4 bytes of a typical age-encrypted file.
    _AGE_MAGIC = b"\x61\x67\x65\x2d\xa6\xf4\x88\xff"

    @pytest.fixture()
    def _mocks(self, tmp_path):
        """Patch TrainingArguments and Trainer to avoid GPU / real HF objects."""
        with (
            patch("paramem.training.trainer.TrainingArguments") as mock_args_cls,
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            mock_args_cls.return_value = MagicMock()
            yield tmp_path

    def _seed_stale_checkpoint(self, output_dir: Path) -> Path:
        """Create a stale ``checkpoint-8/`` with a binary (age-like) README.md.

        This simulates the exact crash scenario: a prior fold wrote an
        age-encrypted checkpoint and then crashed before cleaning up.
        """
        ckpt = output_dir / "checkpoint-8"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "README.md").write_bytes(self._AGE_MAGIC + b"\x00" * 32)
        return ckpt

    def _seed_stale_staging_resume(self, output_dir: Path, fp_dataset: str, fp_config: str) -> Path:
        """Write a ``staging_resume.json`` whose fingerprints do NOT match the run.

        Forces the fresh-start branch in ``train_adapter``.
        """
        stale_state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_dataset + "_STALE",
            "training_config_fingerprint": fp_config + "_STALE",
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": "",
            "checkpoint_path": "",
            "started_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        scratch = output_dir / "staging_resume.json"
        scratch.write_bytes(json.dumps(stale_state, indent=2).encode())
        return scratch

    def test_fresh_start_purges_stale_checkpoint_and_writes_new_marker(self, _mocks, tmp_path):
        """Fingerprint mismatch triggers fresh-start → stale checkpoint-8 removed.

        Assertions:
        1. No exception raised (the age-binary README.md does not crash PEFT).
        2. The stale ``checkpoint-8/`` dir is deleted before training begins.
        3. A fresh ``staging_resume.json`` was written with the current fingerprints
           (verified by a trainer that reads and records its content during train()).

        ``staging_resume.json`` is deleted post-success by ``scratch_path.unlink``,
        so assertion 3 captures the marker state DURING training via a capturing
        trainer subclass.
        """
        _CapturingTrainer.captured_train_kwargs.clear()
        tmp_path = _mocks

        from paramem.training.trainer import _fingerprint_dataset, _fingerprint_training_config

        output_dir = tmp_path / "consolidation_refresh" / "episodic"
        output_dir.mkdir(parents=True)

        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        dataset = _make_dataset()
        tc = _minimal_training_config()
        ac = _minimal_adapter_config()

        # Compute the real fingerprints the run will use.
        from paramem.training.consolidation import ConsolidationLoop

        indexed_ds = ConsolidationLoop._indexed_dataset(dataset)
        fp_dataset = _fingerprint_dataset(indexed_ds)
        fp_config = _fingerprint_training_config(tc, ac)

        # Pre-seed: stale checkpoint with binary content + mismatched marker.
        stale_ckpt = self._seed_stale_checkpoint(output_dir)
        self._seed_stale_staging_resume(output_dir, fp_dataset, fp_config)

        # Verify precondition: the stale checkpoint exists before the call.
        assert stale_ckpt.is_dir(), "Precondition: stale checkpoint-8 must exist"

        # Trainer that records the staging_resume.json content during train().
        # The marker is written before train() is called and deleted post-success,
        # so this is the only window to observe it.
        marker_state_during_train: list[dict] = []

        class _MarkerCapturingTrainer(_CapturingTrainer):
            def train(self, resume_from_checkpoint=None):
                scratch = output_dir / "staging_resume.json"
                if scratch.exists():
                    try:
                        marker_state_during_train.append(json.loads(scratch.read_bytes()))
                    except Exception:
                        pass
                return super().train(resume_from_checkpoint=resume_from_checkpoint)

        # Run train_adapter; must not raise even though README.md has binary bytes.
        read_plain = lambda p: Path(p).read_bytes()  # noqa: E731
        write_plain = lambda p, d: Path(p).write_bytes(d)  # noqa: E731
        with (
            patch("paramem.training.trainer.Trainer", new=_MarkerCapturingTrainer),
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
            patch("paramem.backup.encryption.read_maybe_encrypted", side_effect=read_plain),
            patch("paramem.backup.encryption.write_infra_bytes", side_effect=write_plain),
        ):
            train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=output_dir,
            )

        # Assertion 1: stale checkpoint-8 was removed before training.
        assert not stale_ckpt.exists(), (
            "stale checkpoint-8 must be purged on the fresh-start branch; "
            "leftover encrypted dirs cause PEFT UnicodeDecodeError on save_pretrained"
        )

        # Assertion 2: a fresh staging_resume.json was written with correct fingerprints.
        assert marker_state_during_train, (
            "staging_resume.json must exist and be readable during train() — "
            "the fresh-start branch must write it before calling Trainer"
        )
        written = marker_state_during_train[0]
        assert written["dataset_fingerprint"] == fp_dataset, (
            f"Fresh staging_resume.json must carry the current dataset fingerprint; "
            f"got {written['dataset_fingerprint']!r}"
        )

    def test_resume_branch_does_not_purge_valid_checkpoint(self, _mocks, tmp_path):
        """Matching fingerprints + existing checkpoint dir → resume branch, no purge.

        Scoping lock: the purge must NEVER run on the resume branch, or it would
        delete the checkpoint we are about to resume from.

        This test verifies that when the resume branch is taken, ``trainer.train()``
        is called with ``resume_from_checkpoint`` pointing to the valid checkpoint
        (proving the checkpoint was present at training time, not purged).  The
        post-success ``_clean_scratch`` call legitimately removes all checkpoint-*
        dirs after a normal completion — the assertion window is therefore DURING
        training, captured via a trainer subclass.
        """
        _CapturingTrainer.captured_train_kwargs.clear()
        tmp_path = _mocks

        from paramem.training.trainer import _fingerprint_dataset, _fingerprint_training_config

        output_dir = tmp_path / "consolidation_refresh" / "episodic"
        output_dir.mkdir(parents=True)

        model = _make_peft_model()
        tokenizer = _make_tokenizer()
        dataset = _make_dataset()
        tc = _minimal_training_config()
        ac = _minimal_adapter_config()

        from paramem.training.consolidation import ConsolidationLoop

        indexed_ds = ConsolidationLoop._indexed_dataset(dataset)
        fp_dataset = _fingerprint_dataset(indexed_ds)
        fp_config = _fingerprint_training_config(tc, ac)

        # Create a valid (plaintext) checkpoint directory.
        valid_ckpt = output_dir / "checkpoint-8"
        valid_ckpt.mkdir(parents=True)
        (valid_ckpt / "README.md").write_bytes(b"valid plaintext content")

        # Write a staging_resume.json with MATCHING fingerprints + the checkpoint path.
        matching_state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_dataset,
            "training_config_fingerprint": fp_config,
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": str(valid_ckpt),
            "checkpoint_path": "",
            "started_at": "2026-06-21T00:00:00+00:00",
            "updated_at": "2026-06-21T00:00:00+00:00",
        }
        scratch = output_dir / "staging_resume.json"
        scratch.write_bytes(json.dumps(matching_state, indent=2).encode())

        # Trainer that records whether the checkpoint existed at train() time.
        ckpt_present_during_train: list[bool] = []

        class _ExistenceCapturingTrainer(_CapturingTrainer):
            def train(self, resume_from_checkpoint=None):
                ckpt_present_during_train.append(valid_ckpt.is_dir())
                return super().train(resume_from_checkpoint=resume_from_checkpoint)

        read_plain = lambda p: Path(p).read_bytes()  # noqa: E731
        write_plain = lambda p, d: Path(p).write_bytes(d)  # noqa: E731
        with (
            patch("paramem.training.trainer.Trainer", new=_ExistenceCapturingTrainer),
            patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
            patch("paramem.backup.encryption.read_maybe_encrypted", side_effect=read_plain),
            patch("paramem.backup.encryption.write_infra_bytes", side_effect=write_plain),
        ):
            train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=output_dir,
            )

        # The checkpoint must have been present when train() was called — it was NOT
        # purged by the fresh-start branch (which only runs when _effective_resume is None).
        assert ckpt_present_during_train, "trainer.train() must have been called"
        assert ckpt_present_during_train[0], (
            "valid checkpoint-8 must be present at train() time on the resume branch; "
            "the purge runs ONLY on the fresh-start branch (_effective_resume is None)"
        )

        # Verify resume_from_checkpoint was forwarded (confirms resume branch, not fresh-start).
        assert len(_CapturingTrainer.captured_train_kwargs) == 1
        forwarded = _CapturingTrainer.captured_train_kwargs[0]["resume_from_checkpoint"]
        assert forwarded == str(valid_ckpt), (
            f"On the resume branch, trainer.train() must receive the checkpoint path; "
            f"got {forwarded!r} — if None, the fresh-start branch ran (purge is a bug here)"
        )
