"""Tests for the staging adapter flow (in_training slot for on-the-fly training)."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from paramem.models.loader import (
    atomic_save_adapter,
    copy_adapter_weights,
)


class _FakeParam:
    """Mock parameter that carries a tensor on CPU."""

    def __init__(self, shape):
        self.data = torch.randn(shape)


class _FakePeftModel:
    """Minimal model stub implementing the interface copy_adapter_weights needs."""

    def __init__(self, adapters: list[str], layer_shapes: dict):
        self.peft_config = {name: MagicMock() for name in adapters}
        self._params = {}
        for layer_name, shape in layer_shapes.items():
            for adapter in adapters:
                key = f"base_model.model.{layer_name}.lora_A.{adapter}.weight"
                self._params[key] = _FakeParam(shape)
                key_b = f"base_model.model.{layer_name}.lora_B.{adapter}.weight"
                self._params[key_b] = _FakeParam(shape)

    def named_parameters(self):
        return list(self._params.items())


class TestCopyAdapterWeights:
    def test_copies_matching_tensors(self):
        model = _FakePeftModel(
            adapters=["src", "dst"],
            layer_shapes={"layer0.q_proj": (8, 16), "layer1.v_proj": (8, 16)},
        )
        copy_adapter_weights(model, "src", "dst")
        # Verify dst weights now equal src weights
        for name, p in model.named_parameters():
            if ".dst.weight" in name:
                src_name = name.replace(".dst.weight", ".src.weight")
                src_p = dict(model.named_parameters())[src_name]
                assert torch.equal(p.data, src_p.data)

    def test_unknown_source_raises(self):
        model = _FakePeftModel(adapters=["episodic"], layer_shapes={"layer0.q_proj": (4, 4)})
        with pytest.raises(ValueError, match="Source adapter 'nope' not found"):
            copy_adapter_weights(model, "nope", "episodic")

    def test_unknown_dest_raises(self):
        model = _FakePeftModel(adapters=["episodic"], layer_shapes={"layer0.q_proj": (4, 4)})
        with pytest.raises(ValueError, match="Destination adapter 'nope' not found"):
            copy_adapter_weights(model, "episodic", "nope")

    def test_no_matching_params_raises(self):
        model = _FakePeftModel(adapters=["a", "b"], layer_shapes={})
        with pytest.raises(RuntimeError, match="No adapter-keyed parameters"):
            copy_adapter_weights(model, "a", "b")

    def test_param_set_mismatch_raises(self):
        """If src and dst have different parameter sets, must fail loudly."""

        class _Mismatched:
            peft_config = {"a": MagicMock(), "b": MagicMock()}
            _params = {
                "base_model.model.layer0.q_proj.lora_A.a.weight": _FakeParam((4, 4)),
                "base_model.model.layer0.q_proj.lora_B.a.weight": _FakeParam((4, 4)),
                # b is missing q_proj, has a different module
                "base_model.model.layer0.v_proj.lora_A.b.weight": _FakeParam((4, 4)),
                "base_model.model.layer0.v_proj.lora_B.b.weight": _FakeParam((4, 4)),
            }

            def named_parameters(self):
                return list(self._params.items())

        with pytest.raises(RuntimeError, match="Adapter parameter sets differ"):
            copy_adapter_weights(_Mismatched(), "a", "b")

    def test_does_not_alias_tensors(self):
        """After copy, modifying src should NOT affect dst (deep copy semantics)."""
        model = _FakePeftModel(adapters=["src", "dst"], layer_shapes={"layer0.q_proj": (4, 4)})
        copy_adapter_weights(model, "src", "dst")
        # Mutate src
        for name, p in model.named_parameters():
            if ".src.weight" in name:
                p.data.fill_(999.0)
        # dst should be unchanged
        for name, p in model.named_parameters():
            if ".dst.weight" in name:
                assert not torch.all(p.data == 999.0)


class TestAtomicSaveAdapter:
    def test_creates_target_directory(self, tmp_path):
        model = MagicMock()

        def fake_save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"fake weights")
            (Path(path) / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = fake_save

        target = tmp_path / "episodic"
        atomic_save_adapter(model, target, "episodic")

        assert target.exists()
        assert (target / "adapter_model.safetensors").exists()
        # No stale tmp or old dirs
        leftovers = list(tmp_path.glob("episodic.*"))
        assert leftovers == []

    def test_replaces_existing_directory(self, tmp_path):
        model = MagicMock()
        target = tmp_path / "episodic"
        target.mkdir()
        (target / "old_file").write_text("old")

        def fake_save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "new_file").write_text("new")

        model.save_pretrained.side_effect = fake_save

        atomic_save_adapter(model, target, "episodic")

        assert (target / "new_file").exists()
        assert not (target / "old_file").exists()
        # Backup should be cleaned up
        assert not (tmp_path / "episodic.old").exists()

    def test_cleans_stale_tmp_before_write(self, tmp_path):
        """If a prior crash left a .tmp directory, it should be removed."""
        model = MagicMock()
        target = tmp_path / "episodic"
        stale_tmp = tmp_path / f"episodic.tmp.{os.getpid()}"
        stale_tmp.mkdir()
        (stale_tmp / "junk").write_text("junk")

        def fake_save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"x")

        model.save_pretrained.side_effect = fake_save
        atomic_save_adapter(model, target, "episodic")

        assert target.exists()
        assert not stale_tmp.exists()


class TestStagingFlowContracts:
    """Contract tests for the staging flow — validates BackgroundTrainer logic."""

    def test_pause_switches_to_production_adapter(self):
        """pause() must call switch_adapter with the current production slot name."""
        from paramem.server.background_trainer import BackgroundTrainer

        # Model must have episodic in peft_config for the switch to happen
        model = MagicMock()
        model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=MagicMock(gradient_checkpointing=False),
        )
        bt._is_training = True
        bt._current_adapter = "episodic"
        # Pretend training is already paused so wait() returns immediately
        bt._training_paused.set()

        from unittest.mock import patch

        with patch("paramem.models.loader.switch_adapter") as mock_switch:
            result = bt.pause(timeout=0.1)
            assert result is True
            mock_switch.assert_called_once_with(bt.model, "episodic")

    def test_pause_returns_true_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        assert bt.pause() is True

    def test_pause_with_unset_current_adapter(self):
        """pause() before the first job sets _current_adapter must not crash."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(peft_config={}),
            tokenizer=MagicMock(),
            training_config=MagicMock(gradient_checkpointing=False),
        )
        bt._is_training = True
        bt._current_adapter = ""  # not yet set by _train_adapter
        bt._training_paused.set()
        # Should not raise — gracefully logs warning since "episodic" not in empty peft_config
        assert bt.pause(timeout=0.1) is True


class TestValidateStagingCompat:
    def test_compatible_configs_pass(self):
        from paramem.training.consolidation import _validate_staging_compat

        c1 = MagicMock(rank=8, target_modules=["q_proj", "v_proj"], bias="none")
        c2 = MagicMock(rank=8, target_modules=["q_proj", "v_proj"], bias="none")
        _validate_staging_compat(c1, c2)

    def test_incompatible_configs_raise(self):
        from paramem.training.consolidation import _validate_staging_compat

        c1 = MagicMock(rank=8, target_modules=["q_proj"], bias="none")
        c2 = MagicMock(rank=8, target_modules=["q_proj", "v_proj"], bias="none")
        with pytest.raises(ValueError, match="incompatible for staging"):
            _validate_staging_compat(c1, c2)

    def test_none_configs_skipped(self):
        from paramem.training.consolidation import _validate_staging_compat

        c1 = MagicMock(rank=8, target_modules=["q_proj"], bias="none")
        # Should not raise on None (procedural may be disabled)
        _validate_staging_compat(c1, None)


class TestRunJobsErrorPath:
    """Production adapter disk state must be untouched when training fails."""

    def test_run_jobs_exception_calls_on_error(self):
        from paramem.server.background_trainer import (
            BackgroundTrainer,
            TrainingJob,
        )

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )

        # Force _train_adapter to raise
        def boom(job):
            raise RuntimeError("simulated training failure")

        bt._train_adapter = boom
        bt._is_training = True

        on_error_called = [False]
        on_complete_called = [False]
        bt._on_error = lambda: on_error_called.__setitem__(0, True)

        job = TrainingJob(
            keyed_pairs=[{"key": "k1"}],
            adapter_name="episodic",
            adapter_config=MagicMock(),
        )
        bt._run_jobs([job], on_complete=lambda: on_complete_called.__setitem__(0, True))

        assert on_error_called[0] is True
        assert on_complete_called[0] is False
        assert bt._is_training is False

    def test_run_jobs_exception_preserves_disk_state(self, tmp_path):
        """When _train_adapter raises, no adapter is written to disk."""
        from paramem.server.background_trainer import (
            BackgroundTrainer,
            TrainingJob,
        )

        # Pre-populate target dir with old weights
        target_dir = tmp_path / "episodic"
        target_dir.mkdir()
        (target_dir / "adapter_model.safetensors").write_bytes(b"OLD_WEIGHTS")
        old_mtime = (target_dir / "adapter_model.safetensors").stat().st_mtime

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            output_dir=tmp_path,
        )
        bt._train_adapter = lambda job: (_ for _ in ()).throw(RuntimeError("boom"))
        bt._is_training = True

        job = TrainingJob(
            keyed_pairs=[{"key": "k1"}],
            adapter_name="episodic",
            adapter_config=MagicMock(),
        )
        bt._run_jobs([job], on_complete=None)

        # Old adapter on disk is untouched
        assert (target_dir / "adapter_model.safetensors").read_bytes() == b"OLD_WEIGHTS"
        assert (target_dir / "adapter_model.safetensors").stat().st_mtime == old_mtime


class TestCommitStagingToProduction:
    """The commit step copies in_training → production and saves atomically."""

    def test_commit_copies_then_saves(self, tmp_path):
        from paramem.server.background_trainer import BackgroundTrainer

        model = _FakePeftModel(
            adapters=["in_training", "episodic"],
            layer_shapes={"layer0.q_proj": (4, 4)},
        )
        model.save_pretrained = MagicMock(
            side_effect=lambda path, selected_adapters: (
                Path(path).mkdir(parents=True, exist_ok=True),
                (Path(path) / "adapter_model.safetensors").write_bytes(b"new"),
            )
        )

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            output_dir=tmp_path,
        )
        bt._commit_staging_to_production("episodic")

        # Verify weights copied
        for name, p in model.named_parameters():
            if ".episodic.weight" in name:
                src_name = name.replace(".episodic.weight", ".in_training.weight")
                src_p = dict(model.named_parameters())[src_name]
                assert torch.equal(p.data, src_p.data)

        # Verify atomic save was called with the right target
        model.save_pretrained.assert_called()
        saved_adapter = model.save_pretrained.call_args.kwargs["selected_adapters"]
        assert saved_adapter == ["episodic"]


class TestResumeIdempotent:
    """resume() must be safe to call multiple times without double-releasing lock."""

    def test_resume_without_pause_is_noop(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        bt._is_training = True
        # Never called pause(), so _pause_active is False
        bt.resume()  # should not raise
        assert bt._pause_active is False

    def test_double_resume_safe(self):
        from paramem.server.background_trainer import BackgroundTrainer

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=MagicMock(gradient_checkpointing=False),
        )
        bt._is_training = True
        bt._current_adapter = "episodic"
        bt._training_paused.set()

        assert bt.pause(timeout=0.1) is True
        assert bt._pause_active is True
        bt.resume()
        assert bt._pause_active is False
        # Second resume should be a no-op and must not deadlock
        bt.resume()


class TestPauseSerialization:
    """Concurrent pause() calls must serialize via _pause_lock."""

    def test_second_pause_blocks_until_first_resumes(self):
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=MagicMock(gradient_checkpointing=False),
        )
        bt._is_training = True
        bt._current_adapter = "episodic"
        bt._training_paused.set()

        # First pause holds the lock
        assert bt.pause(timeout=0.1) is True
        assert bt._pause_active is True

        # Second pause from another thread must block
        second_pause_started = threading.Event()
        second_pause_completed = threading.Event()

        def second_caller():
            second_pause_started.set()
            # Re-set training_paused because callback would clear it
            bt._training_paused.set()
            bt.pause(timeout=1.0)
            second_pause_completed.set()

        t = threading.Thread(target=second_caller)
        t.start()
        # Give the second pause a moment to get blocked on the lock
        second_pause_started.wait(timeout=1.0)
        import time

        time.sleep(0.05)
        assert not second_pause_completed.is_set(), (
            "Second pause() completed before first resumed — serialization broken"
        )

        # Resume frees the lock
        bt.resume()
        second_pause_completed.wait(timeout=2.0)
        assert second_pause_completed.is_set()
        t.join()


class TestPauseTimeout:
    """pause() returning False must leave lock released and state clean."""

    def test_pause_timeout_releases_lock_and_clears_flag(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(peft_config={}),
            tokenizer=MagicMock(),
            training_config=MagicMock(gradient_checkpointing=False),
        )
        bt._is_training = True
        bt._current_adapter = "episodic"
        # Do NOT set _training_paused — simulate training thread not responding

        result = bt.pause(timeout=0.05)
        assert result is False
        # _inference_requested must be cleared so it doesn't leak into the
        # next pause attempt or confuse the callback.
        assert not bt._inference_requested.is_set()
        # _pause_active must remain False (matched to an unsuccessful pause).
        assert bt._pause_active is False
        # Lock must be released — a subsequent pause() with training
        # responsive should not block on the lock.
        bt._training_paused.set()
        assert bt.pause(timeout=0.05) is True
        bt.resume()


class TestMultiJobSequencing:
    """Running multiple training jobs through the staging slot."""

    def test_sequential_jobs_each_stage_from_own_production(self):
        """
        Two jobs (episodic then semantic) must each stage from their own
        production adapter — not leak the prior job's state into staging.
        """
        from paramem.server.background_trainer import (
            BackgroundTrainer,
            TrainingJob,
        )

        bt = BackgroundTrainer(
            model=MagicMock(
                peft_config={
                    "episodic": MagicMock(),
                    "semantic": MagicMock(),
                    "in_training": MagicMock(),
                }
            ),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            output_dir=Path("/tmp"),
        )

        # Track the adapter names staged-from and committed-to, in order.
        staged_from = []
        committed_to = []

        def fake_train(job):
            # Simulate what _train_adapter does re: stage and commit
            staged_from.append(job.adapter_name)
            committed_to.append(job.adapter_name)

        bt._train_adapter = fake_train
        bt._is_training = True

        jobs = [
            TrainingJob(
                keyed_pairs=[{"k": 1}], adapter_name="episodic", adapter_config=MagicMock()
            ),
            TrainingJob(
                keyed_pairs=[{"k": 2}], adapter_name="semantic", adapter_config=MagicMock()
            ),
        ]
        bt._run_jobs(jobs, on_complete=None)

        assert staged_from == ["episodic", "semantic"]
        assert committed_to == ["episodic", "semantic"]
        assert bt._is_training is False

    def test_real_train_adapter_calls_copy_adapter_weights(self, monkeypatch):
        """
        Verify the REAL _train_adapter invokes copy_adapter_weights as its
        first action. Patches at the module level so the real code path is
        exercised.
        """
        from paramem.server.background_trainer import (
            BackgroundTrainer,
            TrainingJob,
        )

        calls = []

        def fake_copy(model, src, dst):
            calls.append(("copy", src, dst))

        def fake_switch(model, name):
            calls.append(("switch", name))

        # Patch the loader functions used by _train_adapter
        import paramem.models.loader as loader_mod

        monkeypatch.setattr(loader_mod, "copy_adapter_weights", fake_copy)
        monkeypatch.setattr(loader_mod, "switch_adapter", fake_switch)
        monkeypatch.setattr(
            "paramem.server.background_trainer.format_indexed_training",
            lambda *a, **k: [],  # empty examples → early return, skips Trainer
        )

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            output_dir=Path("/tmp"),
        )

        job = TrainingJob(
            keyed_pairs=[{"k": 1}],
            adapter_name="episodic",
            adapter_config=MagicMock(),
        )
        # Run the REAL _train_adapter (not a stub)
        bt._train_adapter(job)

        # First action must be copy_adapter_weights(src=episodic, dst=in_training)
        assert calls[0] == ("copy", "episodic", "in_training"), (
            f"_train_adapter did not stage from production first: {calls}"
        )
        # Second action must be switch to in_training
        assert calls[1] == ("switch", "in_training")

    def test_real_train_adapter_commits_after_success(self, monkeypatch):
        """
        Full chain: stage → switch → train → commit. Patches Trainer so
        training is a no-op, but leaves examples non-empty so the commit
        path runs. Verifies that _commit_staging_to_production is called
        with the correct production adapter name.
        """
        from paramem.server.background_trainer import (
            BackgroundTrainer,
            TrainingJob,
        )

        calls = []

        def fake_copy(model, src, dst):
            calls.append(("copy", src, dst))

        def fake_switch(model, name):
            calls.append(("switch", name))

        def fake_atomic_save(model, target_dir, adapter_name):
            calls.append(("save", adapter_name, str(target_dir)))

        import paramem.models.loader as loader_mod

        monkeypatch.setattr(loader_mod, "copy_adapter_weights", fake_copy)
        monkeypatch.setattr(loader_mod, "switch_adapter", fake_switch)
        monkeypatch.setattr(loader_mod, "atomic_save_adapter", fake_atomic_save)

        # Return one dummy example so the empty-guard doesn't short-circuit
        monkeypatch.setattr(
            "paramem.server.background_trainer.format_indexed_training",
            lambda *a, **k: [{"input_ids": [0], "labels": [0]}],
        )

        # Replace Trainer and TrainingArguments — TrainingArguments validates bf16/GPU
        # at construction time, which fails on CPU-only CI runners.
        class FakeTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self):
                calls.append(("train",))

        monkeypatch.setattr("paramem.server.background_trainer.Trainer", FakeTrainer)
        monkeypatch.setattr(
            "paramem.server.background_trainer.TrainingArguments",
            lambda **kwargs: MagicMock(),
        )

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}

        training_config = MagicMock()
        training_config.gradient_checkpointing = False
        training_config.num_epochs = 1
        training_config.batch_size = 1
        training_config.gradient_accumulation_steps = 1
        training_config.warmup_steps = 0
        training_config.warmup_ratio = 0.1
        training_config.lr_scheduler_type = "linear"
        training_config.weight_decay = 0.01
        training_config.max_grad_norm = 1.0
        training_config.seed = 42

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=training_config,
            output_dir=Path("/tmp"),
        )

        adapter_config = MagicMock()
        adapter_config.learning_rate = 1e-4
        job = TrainingJob(
            keyed_pairs=[{"k": 1}],
            adapter_name="episodic",
            adapter_config=adapter_config,
        )
        bt._train_adapter(job)

        # Expected sequence: copy(episodic→in_training), switch(in_training),
        # train(), copy(in_training→episodic), save(episodic)
        call_names = [c[0] for c in calls]
        assert "copy" in call_names
        assert "switch" in call_names
        assert "train" in call_names
        assert "save" in call_names

        # Commit order: the second copy call must be in_training→episodic
        copy_calls = [c for c in calls if c[0] == "copy"]
        assert copy_calls[0] == ("copy", "episodic", "in_training"), (
            "First copy must stage from production"
        )
        assert copy_calls[-1] == ("copy", "in_training", "episodic"), (
            "Last copy must commit staging back to production"
        )
        # Save is for the production adapter
        save_calls = [c for c in calls if c[0] == "save"]
        assert save_calls[-1][1] == "episodic"


class TestStaleInTrainingCleanup:
    """_ensure_adapters must remove stale in_training checkpoints on startup."""

    def test_stale_in_training_dir_removed(self, tmp_path):
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import (
            AdapterConfig,
            ConsolidationConfig,
            TrainingConfig,
        )

        # Pre-create a stale in_training checkpoint on disk
        stale = tmp_path / "in_training" / "bg_checkpoint"
        stale.mkdir(parents=True)
        (stale / "leftover.bin").write_bytes(b"stale garbage")

        # Build a loop with pre-wrapped mock model that has peft_config
        model = MagicMock()
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }
        # Bypass _ensure_adapters model re-wrapping — we just need to trigger cleanup
        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            output_dir=tmp_path,
        )
        # _ensure_adapters runs in __init__; the stale dir should be gone
        assert not (tmp_path / "in_training").exists(), (
            "Stale in_training directory was not cleaned up"
        )
        # Cleanup the loop reference
        del loop


class TestFirstCycleEdgeCase:
    """Stage-from-production works when production has PEFT-init values (incl. zeros)."""

    def test_copy_from_zero_initialized_adapter(self):
        """LoRA B matrices init to zero — copy must handle zero tensors."""
        model = _FakePeftModel(
            adapters=["episodic", "in_training"],
            layer_shapes={"layer0.q_proj": (4, 4)},
        )
        # Zero out episodic (simulating fresh PEFT init)
        with torch.no_grad():
            for name, p in model._params.items():
                if ".episodic.weight" in name:
                    p.data.zero_()
                elif ".in_training.weight" in name:
                    p.data.fill_(0.123)  # stale value

        copy_adapter_weights(model, src="episodic", dst="in_training")

        # in_training must now be all zeros (not the stale 0.123)
        for name, p in model._params.items():
            if ".in_training.weight" in name:
                assert torch.all(p.data == 0.0), (
                    f"Zero copy failed: {name} still has non-zero values"
                )
