"""Tests for the staging adapter flow (in_training slot for on-the-fly training)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import PeftModel

from paramem.memory.store import MemoryStore as _MS
from paramem.models.loader import (
    copy_adapter_weights,
    drop_adapter_slot,
)
from paramem.utils.config import AdapterConfig, TrainingConfig


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
            memory_store=_MS(),
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


def _make_train_adapter_model():
    """Fake PeftModel-like object exercising ``train_adapter``'s real
    staging-init decision table (``_ensure_staging_slot`` / ``create_adapter``
    / ``copy_adapter_weights`` / ``drop_adapter_slot`` /
    ``lora_b_frobenius_norm`` all run for real against it) -- no GPU, no real
    PEFT model.  ``__class__ = PeftModel`` (mirrors
    ``tests/test_loader.py::_make_fake_backup_model``) so ``create_adapter``
    takes the ``add_adapter`` path rather than a real ``get_peft_model()``
    wrap.
    """
    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {}
    params: dict = {}

    def _seed(name, lora_b_value=0.0):
        model.peft_config[name] = MagicMock(base_model_name_or_path=None)
        params[f"base_model.model.layer0.q_proj.lora_A.{name}.weight"] = MagicMock(
            data=torch.ones(2, 2)
        )
        params[f"base_model.model.layer0.q_proj.lora_B.{name}.weight"] = MagicMock(
            data=torch.full((2, 2), float(lora_b_value))
        )

    def _add_adapter(name, lora_config):
        # Mirrors real PEFT: a freshly created adapter starts LoRA-B at zero.
        _seed(name, lora_b_value=0.0)

    def _set_adapter(name):
        model.active_adapter = name

    def _delete_adapter(name):
        model.peft_config.pop(name, None)
        for key in [k for k in params if f".{name}." in k]:
            del params[key]

    model.add_adapter.side_effect = _add_adapter
    model.set_adapter.side_effect = _set_adapter
    model.delete_adapter.side_effect = _delete_adapter
    model.named_parameters.side_effect = lambda: list(params.items())
    model.parameters.side_effect = lambda: [p.data for p in params.values()]
    model.get_base_model.return_value.config._name_or_path = "fake/base-model"
    model._params = params
    model._seed = _seed
    return model


def _null_trainer_patches():
    """The three HF-Trainer-boundary patches every real-``train_adapter``
    test in this class needs (mirrors
    ``tests/test_background_trainer.py::TestTrainAdapterAbortReturn``): no
    HF Trainer construction, no ``TrainingArguments`` validation, no real
    checkpoint encryption callback."""

    class _NullTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self, resume_from_checkpoint=None):
            return MagicMock(metrics={"train_loss": 0.1})

    return (
        patch("paramem.training.trainer.ParamemTrainer", new=_NullTrainer),
        patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
        patch(
            "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
            MagicMock,
        ),
    )


class TestTrainAdapterStagingInitTable:
    """``train_adapter``'s own four-way staging-init decision table: the
    staging slot's starting weights are decided once, between
    ``_ensure_staging_slot`` and ``model.set_adapter(STAGING_ADAPTER)`` --
    nothing here ever writes the production tier (*adapter_name*)."""

    def _training_config(self):
        return TrainingConfig(
            num_epochs=1, gradient_checkpointing=False, batch_size=1, warmup_steps=0
        )

    def _dataset(self):
        return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

    def _run(self, model, adapter_config, tmp_path, **kwargs):
        from paramem.training.trainer import STAGING_ADAPTER, train_adapter

        p1, p2, p3 = _null_trainer_patches()
        with p1, p2, p3:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=self._dataset(),
                adapter_name="episodic",
                training_config=self._training_config(),
                adapter_config=adapter_config,
                output_dir=tmp_path,
                **kwargs,
            )
        return metrics, STAGING_ADAPTER

    def test_cold_start_trains_staging_from_lora_zero_without_touching_the_tier(self, tmp_path):
        """warm_start=False on a tier with prior trained (warm) weights:
        staging starts cold, and the tier's own weights never move.  A
        generic ``train_adapter`` parameter -- no production caller passes
        ``False`` (warm start is uniform across every event kind, including
        a reconcile); this pins the boolean's own behaviour directly."""
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=7.0)  # prior trained (warm) weights
        before = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data.clone()

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        metrics, staging = self._run(model, ac, tmp_path, warm_start=False)

        assert metrics["init"] == "cold"
        staging_b = model._params[f"base_model.model.layer0.q_proj.lora_B.{staging}.weight"].data
        assert torch.all(staging_b == 0.0)
        after = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data
        assert torch.equal(before, after)

    def test_warm_start_copies_production_into_staging(self, tmp_path):
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=7.0)
        before = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data.clone()

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        metrics, staging = self._run(model, ac, tmp_path, warm_start=True)

        assert metrics["init"] == "warm"
        staging_b = model._params[f"base_model.model.layer0.q_proj.lora_B.{staging}.weight"].data
        assert torch.all(staging_b == 7.0)
        after = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data
        assert torch.equal(before, after)

    def test_warm_start_wins_over_a_donor_checkpoint_for_a_prior_trained_tier(self, tmp_path):
        """Table precedence: a tier with prior trained weights never seeds
        from a donor, even when one is supplied."""
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=7.0)

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        with patch("paramem.training.donor.load_donor_into_transient_slot") as mock_load:
            metrics, _ = self._run(
                model, ac, tmp_path, warm_start=True, donor_checkpoint_dir=tmp_path / "donor"
            )

        assert metrics["init"] == "warm"
        assert not mock_load.called

    def test_no_prior_weights_no_donor_trains_from_lora_zero(self, tmp_path):
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=0.0)  # resident but cold (never trained)

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        metrics, staging = self._run(
            model, ac, tmp_path, warm_start=True, donor_checkpoint_dir=None
        )

        assert metrics["init"] == "cold"
        staging_b = model._params[f"base_model.model.layer0.q_proj.lora_B.{staging}.weight"].data
        assert torch.all(staging_b == 0.0)

    def test_donor_seeding_initialises_staging_never_the_tier(self, tmp_path):
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=0.0)  # no prior trained weights
        before = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data.clone()

        def _fake_load(model, checkpoint_dir, transient_name):
            model._seed(transient_name, lora_b_value=42.0)

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        with patch("paramem.training.donor.load_donor_into_transient_slot", side_effect=_fake_load):
            metrics, staging = self._run(
                model, ac, tmp_path, warm_start=True, donor_checkpoint_dir=tmp_path / "donor"
            )

        assert metrics["init"] == "donor"
        staging_b = model._params[f"base_model.model.layer0.q_proj.lora_B.{staging}.weight"].data
        assert torch.all(staging_b == 42.0)
        after = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data
        assert torch.equal(before, after)
        assert "_donor_seed" not in model.peft_config, (
            "the transient donor-load slot must be dropped after copy"
        )

    def test_donor_load_failure_degrades_to_cold_and_drops_the_transient(self, tmp_path):
        """Seeding is an optimization over LoRA-zero -- a load/copy failure
        costs the seed, never the fold, and the transient slot is still
        cleaned up."""
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=0.0)

        def _fake_load(model, checkpoint_dir, transient_name):
            model._seed(transient_name, lora_b_value=42.0)

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        with (
            patch("paramem.training.donor.load_donor_into_transient_slot", side_effect=_fake_load),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=RuntimeError("shape mismatch"),
            ),
        ):
            metrics, staging = self._run(
                model, ac, tmp_path, warm_start=True, donor_checkpoint_dir=tmp_path / "donor"
            )

        assert metrics["init"] == "cold"
        assert "_donor_seed" not in model.peft_config

    def test_cold_start_failure_mid_call_leaves_the_tier_untouched(self, tmp_path):
        """An exception raised mid-training (``warm_start=False``) never
        reaches the tier: nothing in the staging-init path writes
        *adapter_name*, so there is nothing for a crash to corrupt there."""
        model = _make_train_adapter_model()
        model._seed("episodic", lora_b_value=7.0)
        before = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data.clone()

        class _RaisingTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                raise RuntimeError("boom")

        ac = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        from paramem.training.trainer import train_adapter

        with (
            patch("paramem.training.trainer.ParamemTrainer", new=_RaisingTrainer),
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=self._dataset(),
                adapter_name="episodic",
                training_config=self._training_config(),
                adapter_config=ac,
                output_dir=tmp_path,
                warm_start=False,
            )

        after = model._params["base_model.model.layer0.q_proj.lora_B.episodic.weight"].data
        assert torch.equal(before, after)
