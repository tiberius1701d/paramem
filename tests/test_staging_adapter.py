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
    drop_adapter_slot,
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


class _FakeActiveModel:
    """Minimal model stub implementing the interface ``drop_adapter_slot`` needs:
    a mutable ``peft_config`` dict, an ``active_adapter`` attribute, and
    recording ``set_adapter``/``delete_adapter`` methods that actually mutate
    state (unlike a bare ``MagicMock``, whose methods are no-ops).

    *fail_switch_to*, when set, makes ``set_adapter`` raise instead of
    switching when called with that name -- simulates PEFT's own switch call
    failing on a broken model state.

    *silent_no_land_switch_to*, when set, makes ``set_adapter`` record the
    call and return normally WITHOUT updating ``active_adapter`` when called
    with that name -- simulates a switch call that does not raise but also
    does not land, distinct from *fail_switch_to*'s raising failure.
    """

    def __init__(
        self,
        adapters: list[str],
        active: "str | None",
        *,
        fail_switch_to=None,
        silent_no_land_switch_to=None,
    ):
        self.peft_config = {name: MagicMock() for name in adapters}
        self.active_adapter = active
        self.set_adapter_calls: list[str] = []
        self.delete_adapter_calls: list[str] = []
        self._fail_switch_to = fail_switch_to
        self._silent_no_land_switch_to = silent_no_land_switch_to

    def set_adapter(self, name):
        self.set_adapter_calls.append(name)
        if name == self._fail_switch_to:
            raise RuntimeError("switch failed")
        if name == self._silent_no_land_switch_to:
            return
        self.active_adapter = name

    def delete_adapter(self, name):
        self.delete_adapter_calls.append(name)
        self.peft_config.pop(name, None)


class TestDropAdapterSlot:
    """Unit coverage for ``paramem.models.loader.drop_adapter_slot`` — the
    one "delete a transient slot" primitive shared by the staging lifecycle
    and the donor build's transient slot."""

    def test_switches_to_fallback_then_deletes_when_target_active_and_fallback_present(self):
        model = _FakeActiveModel(adapters=["in_training", "episodic"], active="in_training")

        drop_adapter_slot(model, "in_training", fallback_adapter="episodic")

        assert model.set_adapter_calls == ["episodic"], (
            "expected a switch to the fallback before delete"
        )
        assert model.delete_adapter_calls == ["in_training"]
        assert "in_training" not in model.peft_config
        assert model.active_adapter == "episodic"

    def test_skips_delete_when_fallback_absent(self):
        """Deleting the model's own ACTIVE adapter with no confirmed
        successor would leave PeftModel with a stale/absent active config --
        the state the project's PEFT rule forbids. With no fallback
        resident, the delete is skipped entirely (not attempted with a
        stale active pointer): the leaked slot stays resident for the
        lifecycle backstop (``assert_staging_absent``) to catch loudly at
        the next training event."""
        model = _FakeActiveModel(adapters=["in_training"], active="in_training")

        drop_adapter_slot(model, "in_training", fallback_adapter="episodic")

        assert model.set_adapter_calls == [], "no switch when the fallback is not resident"
        assert model.delete_adapter_calls == [], "no delete when the fallback is not resident"
        assert "in_training" in model.peft_config
        assert model.active_adapter == "in_training"

    def test_skips_delete_when_fallback_switch_fails(self):
        """A failed switch to the fallback is treated identically to an
        absent fallback: the delete is skipped and the active adapter stays
        exactly where it was, leaving no active-adapter-less window."""
        model = _FakeActiveModel(
            adapters=["in_training", "episodic"], active="in_training", fail_switch_to="episodic"
        )

        drop_adapter_slot(model, "in_training", fallback_adapter="episodic")

        assert model.delete_adapter_calls == [], "no delete when the fallback switch fails"
        assert "in_training" in model.peft_config
        assert model.active_adapter == "in_training"

    def test_skips_delete_when_fallback_switch_does_not_land(self):
        """A switch to the fallback that returns normally WITHOUT actually
        landing (``active_adapter_name`` re-checked after the switch attempt
        still reports the old value, e.g. a broken PEFT internal state that
        no-ops the switch silently) is treated identically to a raised
        switch failure and an absent fallback: the delete is skipped and the
        active adapter stays exactly where it was."""
        model = _FakeActiveModel(
            adapters=["in_training", "episodic"],
            active="in_training",
            silent_no_land_switch_to="episodic",
        )

        drop_adapter_slot(model, "in_training", fallback_adapter="episodic")

        assert model.set_adapter_calls == ["episodic"], "the switch is still attempted"
        assert model.delete_adapter_calls == [], "no delete when the switch does not land"
        assert "in_training" in model.peft_config
        assert model.active_adapter == "in_training"

    def test_noop_when_target_absent(self):
        model = _FakeActiveModel(adapters=["episodic"], active="episodic")

        drop_adapter_slot(model, "in_training", fallback_adapter="episodic")

        assert model.set_adapter_calls == []
        assert model.delete_adapter_calls == []
        assert set(model.peft_config) == {"episodic"}

    def test_deletes_without_switch_when_target_not_active(self):
        """When *name* is not the currently active adapter, no switch is
        needed -- the delete proceeds unconditionally, even with no
        fallback resident at all."""
        model = _FakeActiveModel(adapters=["in_training", "episodic"], active="episodic")

        drop_adapter_slot(model, "in_training", fallback_adapter="semantic")

        assert model.set_adapter_calls == [], "no switch needed when the target is not active"
        assert model.delete_adapter_calls == ["in_training"]
        assert "in_training" not in model.peft_config
        assert model.active_adapter == "episodic"


class TestAtomicSaveAdapter:
    def test_creates_target_directory(self, tmp_path):
        """atomic_save_adapter creates target_dir and a timestamped slot under it.

        Under the timestamped-slot layout: adapter files live in target_dir/<ts>/,
        not in target_dir itself.  Verify the slot exists and contains the adapter files.
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
        alongside the new slot.  Retention/pruning happens separately via the pruner.
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


class TestStagedWeightsDisposalGuard:
    """``paramem.training.trainer.staged_weights``' ``finally`` is the ONE
    disposal designed to run with an in-flight exception (the refusal path
    -- ``RecallGateRejected`` raised from inside the ``with`` body) -- a
    failure disposing of the staging slot must never replace that in-flight
    exception, or the caller's compensation and the incident detail it
    carries would both be lost."""

    def test_disposal_failure_never_replaces_an_in_flight_exception(self):
        from paramem.training.trainer import staged_weights

        model = MagicMock()

        class _Verdict(RuntimeError):
            pass

        with patch(
            "paramem.models.loader.drop_adapter_slot",
            side_effect=RuntimeError("disposal boom"),
        ):
            with pytest.raises(_Verdict, match="original verdict"):
                with staged_weights(model, fallback_adapter="episodic"):
                    raise _Verdict("original verdict")

    def test_disposal_failure_is_logged(self, caplog):
        import logging

        from paramem.training.trainer import staged_weights

        model = MagicMock()

        with patch(
            "paramem.models.loader.drop_adapter_slot",
            side_effect=RuntimeError("disposal boom"),
        ):
            with caplog.at_level(logging.ERROR, logger="paramem.training.trainer"):
                with pytest.raises(RuntimeError, match="original verdict"):
                    with staged_weights(model, fallback_adapter="episodic"):
                        raise RuntimeError("original verdict")

        assert any("drop_adapter_slot" in record.getMessage() for record in caplog.records), (
            "expected the disposal failure to be logged, not silently swallowed"
        )

    def test_disposal_still_runs_on_success(self):
        from paramem.training.trainer import STAGING_ADAPTER, staged_weights

        model = MagicMock()

        with patch("paramem.models.loader.drop_adapter_slot") as mock_drop:
            with staged_weights(model, fallback_adapter="episodic"):
                pass

        mock_drop.assert_called_once_with(model, STAGING_ADAPTER, fallback_adapter="episodic")


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


class TestStaleInTrainingCleanup:
    """ensure_adapters must remove stale in_training checkpoints on startup."""

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
        # __class__ = PeftModel so ensure_adapters' isinstance check
        # short-circuits without restricting the mock's attribute surface.
        model = MagicMock()
        model.__class__ = PeftModel
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }
        # Bypass ensure_adapters model re-wrapping — we just need to trigger cleanup
        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(replay_enabled=False),
            output_dir=tmp_path,
            extraction_scrub={"person name"},
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )
        # ensure_adapters runs in __init__; the stale dir should be gone
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
