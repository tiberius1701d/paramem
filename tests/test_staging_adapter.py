"""Tests for the staging adapter flow (in_training slot for on-the-fly training)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import PeftModel

from paramem.memory.store import MemoryStore as _MS
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
        """atomic_save_adapter creates target_dir and a timestamped slot under it.

        Post-Slice-3a: adapter files live in target_dir/<ts>/, not in target_dir
        itself.  Verify the slot exists and contains the adapter files.
        """
        model = MagicMock()

        def fake_save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"fake weights")
            (Path(path) / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = fake_save

        target = tmp_path / "episodic"
        slot = atomic_save_adapter(model, target, "episodic")

        # target_dir itself exists
        assert target.exists()
        # Slot is a direct child of target_dir
        assert slot.parent == target
        assert (slot / "adapter_model.safetensors").exists()
        # No stale tmp or old dirs
        leftovers = list(tmp_path.glob("episodic.*"))
        assert leftovers == []

    def test_replaces_existing_directory(self, tmp_path):
        """Slot-dir layout preserves history — old slot is NOT removed.

        Semantic change from the old flat layout: the old slot is retained
        alongside the new slot.  Retention/pruning happens separately (Slice 3b).
        The new slot must contain the new file; both slots are visible.
        """
        model = MagicMock()
        target = tmp_path / "episodic"

        # Create a pre-existing slot that looks like an old save
        old_slot = target / "20260420-000000"
        old_slot.mkdir(parents=True)
        (old_slot / "old_file").write_text("old")

        def fake_save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "new_file").write_text("new")

        model.save_pretrained.side_effect = fake_save

        new_slot = atomic_save_adapter(model, target, "episodic")

        # New slot has the new file
        assert (new_slot / "new_file").exists()
        # Old slot is RETAINED (slot-dir preserves history)
        assert (old_slot / "old_file").exists()
        # No .old backup dirs (old codepath gone)
        assert not (tmp_path / "episodic.old").exists()

    def test_cleans_stale_tmp_before_write(self, tmp_path):
        """Stale .pending/<ts>/ dirs from prior crashes are cleaned by sweep_orphan_pending.

        The old .tmp.{pid} codepath is gone; .pending/<ts>/ is the new staging
        area.  Verify that a stale pending slot does not prevent a new save and
        that sweep_orphan_pending removes it.
        """
        from paramem.backup.backup import sweep_orphan_pending

        model = MagicMock()
        target = tmp_path / "episodic"
        target.mkdir()

        # Simulate a stale pending slot from a prior crash
        stale_pending = target / ".pending" / "20260420-120000"
        stale_pending.mkdir(parents=True)
        (stale_pending / "junk").write_text("junk")

        # sweep_orphan_pending should remove the stale slot
        sweep_orphan_pending(target)
        assert not stale_pending.exists()

        def fake_save(path, selected_adapters):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"x")

        model.save_pretrained.side_effect = fake_save
        slot = atomic_save_adapter(model, target, "episodic")

        assert slot.exists()
        assert (slot / "adapter_model.safetensors").exists()

    def test_peft_nested_subdir_flatten_inside_pending_slot(self, tmp_path):
        """Verify PEFT-nested subdir is flattened INSIDE .pending before outer rename.

        This guards against: (1) flattening too early (before save_pretrained),
        (2) flattening too late (after outer rename).  The flatten must happen
        at step 3 of the six-step sequence, inside the pending slot.
        """
        pending_state: dict = {}
        original_rename = Path.rename

        def _intercept_rename(self_path, target_):
            # Intercept only the outer rename from .pending/<ts>/ to final slot
            if (
                ".pending" in str(self_path)
                and self_path.is_dir()
                and self_path.parent.name == ".pending"
            ):
                pending_state["files"] = [f.name for f in self_path.iterdir()]
                pending_state["has_nested"] = (self_path / "episodic").exists()
            return original_rename(self_path, target_)

        def fake_save_nested(path, selected_adapters):
            nested = Path(path) / "episodic"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "adapter_model.safetensors").write_bytes(b"weights")
            (nested / "adapter_config.json").write_text("{}")

        model = MagicMock()
        model.save_pretrained.side_effect = fake_save_nested

        with patch.object(Path, "rename", _intercept_rename):
            final_slot = atomic_save_adapter(model, tmp_path / "episodic", "episodic")

        # Flatten must have run before outer rename
        assert not pending_state.get("has_nested", True), (
            "Nested subdir must be absent inside pending slot at rename time"
        )
        assert "adapter_model.safetensors" in pending_state.get("files", [])

        # Final slot must also be flat
        assert (final_slot / "adapter_model.safetensors").exists()
        assert not (final_slot / "episodic").exists()


class TestStagingFlowContracts:
    """Contract tests for the staging flow — validates BackgroundTrainer logic."""

    def test_abort_returns_false_when_idle(self):
        """abort_for_inference() returns False immediately when no job is active."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        assert bt._active_abort is None
        result = bt.abort_for_inference(timeout=0.01)
        assert result is False

    def test_abort_for_inference_sets_abort_and_quiesces(self):
        """abort_for_inference() sets abort event and returns True when quiesced fires."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        def _fire():
            import time

            time.sleep(0.05)
            bt._active_quiesced.set()

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        result = bt.abort_for_inference(timeout=2.0)
        assert result is True
        assert bt._active_abort.is_set()
        t.join(timeout=1.0)

    def test_abort_with_preload_cache_false_switches_adapter(self):
        """Abort branch calls switch_adapter when preload_cache=False."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
        from paramem.utils.config import AdapterConfig, TrainingConfig

        model = MagicMock()
        model.peft_config = {
            "episodic": MagicMock(target_modules=["q_proj"], r=4),
            "in_training": MagicMock(target_modules=["q_proj"], r=4),
        }
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=TrainingConfig(
                num_epochs=1, gradient_checkpointing=False, batch_size=1
            ),
            output_dir="/tmp/test_abort_switch",
            preload_cache=False,
        )
        job = TrainingJob(
            entries=[{"key": "g1", "subject": "S", "predicate": "p", "object": "O"}],
            adapter_name="episodic",
            adapter_config=AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"]),
        )

        # Pre-install and set abort so _train_adapter_body takes the abort branch.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()
        bt._active_abort.set()

        class _NullTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                return MagicMock(metrics={"train_loss": 0.0})

        switch_calls = []

        def _spy_switch(m, name):
            switch_calls.append(name)

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter", side_effect=_spy_switch),
            patch(
                "paramem.server.background_trainer.format_entry_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch(
                "paramem.server.background_trainer._ensure_staging_shape_matches",
                side_effect=lambda m, _c: m,
            ),
            patch("paramem.training.trainer.Trainer", new=_NullTrainer),
            patch(
                "paramem.training.trainer.TrainingArguments",
                return_value=MagicMock(num_train_epochs=1, output_dir="/tmp"),
            ),
            patch(
                "paramem.server.background_trainer._read_resume_state",
                return_value=None,
            ),
        ):
            bt._train_adapter(job)

        # In the abort branch with preload_cache=False, switch_adapter must
        # restore the production adapter after training stops.
        assert "episodic" in switch_calls, (
            f"Abort branch must call switch_adapter('episodic') when preload_cache=False. "
            f"Got: {switch_calls}"
        )


class TestEnsureStagingShapeMatches:
    """Lazy-rebuild of the in_training staging slot when shape diverges.

    Replaces the prior TestValidateStagingCompat suite — the validator is
    gone (the staging architecture no longer requires uniform shapes
    across production tiers; rebuild on tier-shape transition handles the
    heterogeneous case).
    """

    def _make_model_with_staging(self, target_modules, rank=8):
        """Build a stub PeftModel whose in_training slot has the given shape."""
        peft_config = {"in_training": MagicMock(target_modules=list(target_modules), r=rank)}
        return MagicMock(peft_config=peft_config)

    def test_same_shape_is_noop(self, monkeypatch):
        from paramem.server.background_trainer import _ensure_staging_shape_matches

        called = []
        monkeypatch.setattr(
            "paramem.models.loader.create_adapter",
            lambda *a, **kw: called.append("create") or a[0],
        )
        model = self._make_model_with_staging(["q_proj", "v_proj"])
        target = MagicMock(target_modules=["q_proj", "v_proj"], rank=8)

        result = _ensure_staging_shape_matches(model, target)
        assert result is model
        assert called == [], "no rebuild expected when shapes match"
        model.delete_adapter.assert_not_called()

    def test_target_modules_mismatch_triggers_rebuild(self, monkeypatch):
        from paramem.server.background_trainer import _ensure_staging_shape_matches

        rebuilt_with = []

        def _stub_create(model, cfg, name):
            rebuilt_with.append((cfg.target_modules, name))
            return model

        monkeypatch.setattr("paramem.models.loader.create_adapter", _stub_create)
        attn_only = ["q_proj", "v_proj", "k_proj", "o_proj"]
        attn_mlp = [*attn_only, "gate_proj", "up_proj", "down_proj"]
        model = self._make_model_with_staging(attn_only)
        target = MagicMock(target_modules=attn_mlp, rank=8)

        _ensure_staging_shape_matches(model, target)
        model.delete_adapter.assert_called_once_with("in_training")
        assert rebuilt_with == [(target.target_modules, "in_training")]

    def test_rank_mismatch_triggers_rebuild(self, monkeypatch):
        from paramem.server.background_trainer import _ensure_staging_shape_matches

        rebuilt = []
        monkeypatch.setattr(
            "paramem.models.loader.create_adapter",
            lambda m, cfg, name: rebuilt.append((cfg.rank, name)) or m,
        )
        model = self._make_model_with_staging(["q_proj", "v_proj"], rank=8)
        target = MagicMock(target_modules=["q_proj", "v_proj"], rank=16)

        _ensure_staging_shape_matches(model, target)
        model.delete_adapter.assert_called_once_with("in_training")
        assert rebuilt == [(16, "in_training")]

    def test_module_order_irrelevant(self, monkeypatch):
        """Module order must not matter — comparison is on sorted tuples."""
        from paramem.server.background_trainer import _ensure_staging_shape_matches

        called = []
        monkeypatch.setattr(
            "paramem.models.loader.create_adapter",
            lambda *a, **kw: called.append("create") or a[0],
        )
        model = self._make_model_with_staging(["o_proj", "v_proj", "q_proj", "k_proj"])
        target = MagicMock(target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], rank=8)

        _ensure_staging_shape_matches(model, target)
        assert called == [], "rebuild not expected when modules differ only in order"


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
            entries=[{"key": "k1"}],
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
            entries=[{"key": "k1"}],
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
        # Patch build_manifest_for so this weight-copy / save-path test is
        # not coupled to the manifest serialization path.
        with patch("paramem.adapters.manifest.build_manifest_for", return_value=None):
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


class TestAbortEventLifecycle:
    """Per-job abort events are created, used, and cleared cleanly."""

    def test_abort_event_none_when_idle(self):
        """_active_abort is None when no job is running."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        assert bt._active_abort is None
        assert bt._active_quiesced is None

    def test_abort_event_installed_and_cleared(self):
        """Per-job events are installed before and cleared after _run_callable_queue runs a job."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        job_ran = threading.Event()

        def job():
            # Inside the job, _active_abort must be set.
            assert bt._active_abort is not None
            job_ran.set()

        from contextlib import contextmanager

        @contextmanager
        def _noop_lock():
            yield

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_lock):
            bt.submit(job)
            job_ran.wait(timeout=5.0)

        # After the job, events are cleared in the finally block.
        # Give the thread a moment to reach finally.
        import time

        time.sleep(0.05)
        assert bt._active_abort is None
        assert bt._active_quiesced is None


class TestAbortSignalPropagation:
    """abort_for_inference() signal propagates through training_hooks_for_job."""

    def test_abort_signal_reaches_shutdown_predicate(self):
        """After abort.set(), training_hooks_for_job predicate returns True."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        hooks = bt.training_hooks_for_job()
        assert hooks.on_shutdown_check() is False
        bt._active_abort.set()
        assert hooks.on_shutdown_check() is True

    def test_per_job_closure_does_not_bleed_across_jobs(self):
        """Hooks captured for job A do not pick up job B's abort event."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()
        hooks_a = bt.training_hooks_for_job()

        # Clear (job A done) and install job B.
        with bt._active_state_lock:
            bt._active_abort = None
            bt._active_quiesced = None
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()
        hooks_b = bt.training_hooks_for_job()

        # Set job B's abort.
        bt._active_abort.set()
        # hooks_a must remain False — it captured job A's event (which was never set).
        assert hooks_a.on_shutdown_check() is False
        # hooks_b picks up job B's event.
        assert hooks_b.on_shutdown_check() is True


class TestAbortTimeout:
    """abort_for_inference() timeout returns False cleanly without state leak."""

    def test_abort_timeout_returns_false(self):
        """abort_for_inference() returns False when quiesced does not fire within timeout."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        # Install events but do NOT fire quiesced — simulate slow training.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        result = bt.abort_for_inference(timeout=0.05)
        assert result is False
        # abort flag is set (we still signalled it), but quiesced timed out.
        assert bt._active_abort.is_set()

    def test_abort_timeout_does_not_prevent_subsequent_abort(self):
        """After a timed-out abort, a fresh job's abort_for_inference works normally."""
        import threading

        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(), tokenizer=MagicMock(), training_config=MagicMock()
        )
        # First abort times out (no quiesced fired).
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()
        bt.abort_for_inference(timeout=0.01)

        # Simulate next job — install fresh events.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        def _fire():
            import time

            time.sleep(0.05)
            bt._active_quiesced.set()

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        result = bt.abort_for_inference(timeout=2.0)
        assert result is True
        t.join(timeout=1.0)


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
        # Stub thermal cooldown — it reads real GPU temp and sleeps, so the
        # test would stall on a warm GPU. Not the subject under test.
        bt._cooldown_between_jobs = lambda: None
        bt._is_training = True

        jobs = [
            TrainingJob(entries=[{"k": 1}], adapter_name="episodic", adapter_config=MagicMock()),
            TrainingJob(entries=[{"k": 2}], adapter_name="semantic", adapter_config=MagicMock()),
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
            "paramem.server.background_trainer.format_entry_training",
            lambda *a, **k: [],  # empty examples → early return, skips Trainer
        )

        # Configure peft_config["in_training"] with a concrete target_modules list +
        # rank so _ensure_staging_shape_matches sees a matching shape and skips
        # the rebuild (the rebuild path is exercised in TestEnsureStagingShapeMatches).
        attn_only = ["q_proj", "v_proj", "k_proj", "o_proj"]
        model = MagicMock()
        in_training_cfg = MagicMock(target_modules=list(attn_only), r=8)
        model.peft_config = {"episodic": MagicMock(), "in_training": in_training_cfg}

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            output_dir=Path("/tmp"),
        )

        job = TrainingJob(
            entries=[{"k": 1}],
            adapter_name="episodic",
            adapter_config=MagicMock(target_modules=list(attn_only), rank=8),
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

        def fake_atomic_save(model, target_dir, adapter_name, **kwargs):
            calls.append(("save", adapter_name, str(target_dir)))

        import paramem.models.loader as loader_mod

        monkeypatch.setattr(loader_mod, "copy_adapter_weights", fake_copy)
        monkeypatch.setattr(loader_mod, "switch_adapter", fake_switch)
        monkeypatch.setattr(loader_mod, "atomic_save_adapter", fake_atomic_save)

        # Return one dummy example so the empty-guard doesn't short-circuit
        monkeypatch.setattr(
            "paramem.server.background_trainer.format_entry_training",
            lambda *a, **k: [{"input_ids": [0], "labels": [0]}],
        )

        # BG trainer's _train_adapter now delegates to paramem.training.trainer.
        # train_adapter (Commit 2 refactor, 2026-05-07).  Mock the delegate at
        # BG's import site so the train call is recorded without constructing
        # an HF Trainer or TrainingArguments — the latter validates bf16/GPU at
        # init and fails on CPU-only CI runners.
        def _fake_train_adapter(**kwargs):
            calls.append(("train",))
            return {"train_loss": 0.0}

        monkeypatch.setattr(
            "paramem.server.background_trainer.train_adapter",
            _fake_train_adapter,
        )

        # in_training config carries concrete shape so _ensure_staging_shape_matches
        # is a no-op (the rebuild path has its own dedicated tests).
        attn_only = ["q_proj", "v_proj", "k_proj", "o_proj"]
        model = MagicMock()
        in_training_cfg = MagicMock(target_modules=list(attn_only), r=8)
        model.peft_config = {"episodic": MagicMock(), "in_training": in_training_cfg}

        # Real TrainingConfig instance — BG path calls dataclasses.replace
        # on it (Commit 2 refactor) to override logging_steps before delegating
        # to train_adapter, which requires a genuine dataclass.
        from paramem.utils.config import TrainingConfig

        training_config = TrainingConfig(
            gradient_checkpointing=False,
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
        )

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=training_config,
            output_dir=Path("/tmp"),
        )

        adapter_config = MagicMock(target_modules=list(attn_only), rank=8)
        adapter_config.learning_rate = 1e-4
        job = TrainingJob(
            entries=[{"k": 1}],
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

        # Build a loop with pre-wrapped mock model that has peft_config.
        # __class__ = PeftModel so _ensure_adapters' isinstance check
        # short-circuits without restricting the mock's attribute surface.
        model = MagicMock()
        model.__class__ = PeftModel
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
            memory_store=_MS(replay_enabled=False),
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
