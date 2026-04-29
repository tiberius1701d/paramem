"""Unit tests for BackgroundTrainer epoch-level resume.

Covers:
  - Fresh start writes resume_state.json at each epoch boundary.
  - Final state is removed after successful commit.
  - Simulated restart with matching fingerprint: trainer.train called with
    resume_from_checkpoint; copy_adapter_weights NOT called for staging.
  - Simulated restart with mismatching fingerprint: stale state wiped, fresh
    training starts (copy_adapter_weights IS called).
  - On training exception: resume_state.json remains on disk.
  - Successful completion wipes both resume_state.json and bg_checkpoint dir.

No GPU required — Trainer.train is mocked throughout.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.background_trainer import (
    _RESUME_STATE_FILE,
    BackgroundTrainer,
    TrainingJob,
    _fingerprint_keyed_pairs,
    _fingerprint_training_config,
    _read_resume_state,
    _write_resume_state_atomic,
)
from paramem.utils.config import AdapterConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_model(*adapter_names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel.

    Each adapter mock carries concrete ``target_modules`` and ``r`` so the
    BG-trainer's lazy-rebuild check (``_ensure_staging_shape_matches``) sees
    a shape matching ``_minimal_adapter_config()`` and skips the rebuild
    path. Tests that actually need to exercise rebuild override these
    attributes per-test.
    """
    model = MagicMock()
    model.peft_config = {
        name: MagicMock(target_modules=["q_proj"], r=4) for name in adapter_names
    }
    return model


def _minimal_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=3,
        gradient_checkpointing=False,
        batch_size=1,
        warmup_steps=0,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
    )


def _minimal_adapter_config() -> AdapterConfig:
    return AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])


def _make_bt(tmp_path: Path) -> BackgroundTrainer:
    """Create a BackgroundTrainer wired to tmp_path."""
    model = _make_stub_model("episodic", "in_training")
    config = _minimal_training_config()
    return BackgroundTrainer(
        model=model,
        tokenizer=MagicMock(),
        training_config=config,
        output_dir=tmp_path,
    )


def _make_job(keyed_pairs: list[dict] | None = None) -> TrainingJob:
    return TrainingJob(
        keyed_pairs=keyed_pairs or [{"key": "graph1", "question": "Q?", "answer": "A."}],
        adapter_name="episodic",
        adapter_config=_minimal_adapter_config(),
        inference_fallback_adapter="episodic",
    )


def _fake_trainer_class(epoch_callback_capture: list | None = None, raise_after: int = -1):
    """Return a FakeTrainer class that drives the epoch callback on every epoch.

    Args:
        epoch_callback_capture: If provided, each completed epoch number is
            appended here so tests can verify callback sequencing.
        raise_after: If >= 0, raise RuntimeError after this many epochs.
    """

    class FakeTrainer:
        def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
            self._callbacks = callbacks
            self._args = args

        def train(self, resume_from_checkpoint=None):
            """Fire on_epoch_end callbacks for each epoch, then optionally raise."""
            num_epochs = int(self._args.num_train_epochs)
            for epoch in range(1, num_epochs + 1):
                state = MagicMock()
                state.epoch = epoch
                state.global_step = epoch * 10
                control = MagicMock()
                for cb in self._callbacks:
                    cb.on_epoch_end(self._args, state, control)
                if epoch_callback_capture is not None:
                    epoch_callback_capture.append(epoch)
                if raise_after >= 0 and epoch > raise_after:
                    raise RuntimeError("simulated training failure")

    return FakeTrainer


def _fake_training_arguments(**kwargs):
    """Return a MagicMock that keeps num_train_epochs and output_dir accessible."""
    args = MagicMock()
    args.num_train_epochs = kwargs.get("num_train_epochs", 3)
    args.output_dir = kwargs.get("output_dir", "/tmp/bg_checkpoint")
    return args


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


class TestFingerprintHelpers:
    def test_keyed_pairs_fingerprint_deterministic(self) -> None:
        pairs = [{"key": "g1", "question": "Q?", "answer": "A."}]
        fp1 = _fingerprint_keyed_pairs(pairs)
        fp2 = _fingerprint_keyed_pairs(pairs)
        assert fp1 == fp2

    def test_keyed_pairs_fingerprint_differs_on_change(self) -> None:
        pairs_a = [{"key": "g1", "question": "Q?", "answer": "A."}]
        pairs_b = [{"key": "g1", "question": "Q?", "answer": "B."}]
        assert _fingerprint_keyed_pairs(pairs_a) != _fingerprint_keyed_pairs(pairs_b)

    def test_config_fingerprint_deterministic(self) -> None:
        tc = _minimal_training_config()
        ac = _minimal_adapter_config()
        fp1 = _fingerprint_training_config(tc, ac)
        fp2 = _fingerprint_training_config(tc, ac)
        assert fp1 == fp2

    def test_config_fingerprint_differs_on_epoch_change(self) -> None:
        tc_a = TrainingConfig(num_epochs=3, gradient_checkpointing=False, batch_size=1)
        tc_b = TrainingConfig(num_epochs=30, gradient_checkpointing=False, batch_size=1)
        ac = _minimal_adapter_config()
        assert _fingerprint_training_config(tc_a, ac) != _fingerprint_training_config(tc_b, ac)


# ---------------------------------------------------------------------------
# Atomic write / read helpers
# ---------------------------------------------------------------------------


class TestResumeStateIO:
    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        state = {"adapter_name": "episodic", "last_completed_epoch": 5}
        path = tmp_path / "resume_state.json"
        _write_resume_state_atomic(path, state)
        assert path.exists()
        loaded = _read_resume_state(path)
        assert loaded == state

    def test_read_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = _read_resume_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_read_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "resume_state.json"
        p.write_text("not valid json {{{")
        result = _read_resume_state(p)
        assert result is None

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "resume_state.json"
        _write_resume_state_atomic(path, {"x": 1})
        assert path.exists()

    def test_no_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        """A failed write must not leave a .tmp file behind."""
        path = tmp_path / "resume_state.json"
        bad_state = object()  # not JSON-serialisable
        with pytest.raises(Exception):
            _write_resume_state_atomic(path, bad_state)
        leftover = list(tmp_path.glob("*.tmp.*"))
        assert leftover == [], f"Stale tmp file left after failed write: {leftover}"


# ---------------------------------------------------------------------------
# Fresh start: resume state written per epoch, cleared on success
# ---------------------------------------------------------------------------


class TestFreshStartResumeState:
    """Fresh training (no prior state) writes state after each epoch, clears on success."""

    def test_resume_state_written_after_each_epoch(self, tmp_path: Path) -> None:
        """_write_resume_state is called once per completed epoch."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        write_calls: list[int] = []

        original_write = bt._write_resume_state

        def _spy_write(epoch: int, checkpoint_dir: str) -> None:
            write_calls.append(epoch)
            # Create a fake checkpoint dir so the state is considered valid.
            ckpt = Path(checkpoint_dir) / f"checkpoint-{epoch * 10}"
            ckpt.mkdir(parents=True, exist_ok=True)
            original_write(epoch, checkpoint_dir)

        bt._write_resume_state = _spy_write

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch(
                "paramem.server.background_trainer.Trainer",
                new=_fake_trainer_class(),
            ),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert write_calls == [1, 2, 3], (
            f"Expected _write_resume_state called for epochs [1,2,3], got {write_calls}"
        )

    def test_resume_state_cleared_after_success(self, tmp_path: Path) -> None:
        """resume_state.json is removed after successful commit."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        state_path = bt._resume_state_path()

        def _spy_write(epoch: int, checkpoint_dir: str) -> None:
            ckpt = Path(checkpoint_dir) / f"checkpoint-{epoch * 10}"
            ckpt.mkdir(parents=True, exist_ok=True)
            # Write a real state file so we can assert it gets cleaned up.
            state = {
                "adapter_name": "episodic",
                "last_completed_epoch": epoch,
                "checkpoint_path": str(ckpt),
                "total_epochs": 3,
            }
            _write_resume_state_atomic(state_path, state)

        bt._write_resume_state = _spy_write

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch(
                "paramem.server.background_trainer.Trainer",
                new=_fake_trainer_class(),
            ),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert not state_path.exists(), (
            "resume_state.json must be removed after successful training commit"
        )

    def test_bg_checkpoint_dir_wiped_after_success(self, tmp_path: Path) -> None:
        """bg_checkpoint/ is removed after successful commit."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        checkpoint_root = tmp_path / "in_training" / "bg_checkpoint"

        def _spy_write(epoch: int, checkpoint_dir: str) -> None:
            ckpt = Path(checkpoint_dir) / f"checkpoint-{epoch * 10}"
            ckpt.mkdir(parents=True, exist_ok=True)

        bt._write_resume_state = _spy_write

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch(
                "paramem.server.background_trainer.Trainer",
                new=_fake_trainer_class(),
            ),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert not checkpoint_root.exists(), "bg_checkpoint/ must be wiped after successful commit"

    def test_copy_adapter_weights_called_for_fresh_start(self, tmp_path: Path) -> None:
        """copy_adapter_weights(src=episodic, dst=in_training) is called on fresh start."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        bt._write_resume_state = MagicMock()

        copy_calls: list[tuple] = []

        def _spy_copy(model, src, dst):
            copy_calls.append((src, dst))

        with (
            patch("paramem.models.loader.copy_adapter_weights", side_effect=_spy_copy),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch(
                "paramem.server.background_trainer.Trainer",
                new=_fake_trainer_class(),
            ),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        staging_copy = ("episodic", "in_training")
        assert staging_copy in copy_calls, (
            f"Expected copy_adapter_weights(episodic→in_training) on fresh start. Got: {copy_calls}"
        )


# ---------------------------------------------------------------------------
# Resume with matching fingerprint
# ---------------------------------------------------------------------------


class TestResumeMatchingFingerprint:
    """When fingerprints match and checkpoint exists, resume from that checkpoint."""

    def _plant_resume_state(
        self, tmp_path: Path, bt: BackgroundTrainer, job: TrainingJob, epoch: int
    ) -> Path:
        """Write a resume state file and a matching checkpoint dir."""
        from paramem.server.background_trainer import (
            _fingerprint_keyed_pairs,
            _fingerprint_training_config,
        )

        checkpoint_dir = tmp_path / "in_training" / "bg_checkpoint" / f"checkpoint-{epoch * 10}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "adapter_name": job.adapter_name,
            "inference_fallback_adapter": job.inference_fallback_adapter,
            "training_config_fingerprint": _fingerprint_training_config(
                bt.training_config, job.adapter_config
            ),
            "keyed_pairs_fingerprint": _fingerprint_keyed_pairs(job.keyed_pairs),
            "total_epochs": bt.training_config.num_epochs,
            "last_completed_epoch": epoch,
            "checkpoint_path": str(checkpoint_dir),
            "started_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        state_path = tmp_path / "in_training" / _RESUME_STATE_FILE
        _write_resume_state_atomic(state_path, state)
        return state_path

    def test_train_called_with_resume_from_checkpoint(self, tmp_path: Path) -> None:
        """trainer.train(resume_from_checkpoint=<path>) when fingerprints match."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        self._plant_resume_state(tmp_path, bt, job, epoch=1)
        expected_ckpt = str(tmp_path / "in_training" / "bg_checkpoint" / "checkpoint-10")

        train_kwargs: list[dict] = []

        class CapturingTrainer:
            def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
                self._callbacks = callbacks
                self._args = args

            def train(self, resume_from_checkpoint=None):
                train_kwargs.append({"resume_from_checkpoint": resume_from_checkpoint})
                # Fire epoch callbacks so _write_resume_state is exercised
                for epoch in range(1, int(self._args.num_train_epochs) + 1):
                    state = MagicMock()
                    state.epoch = epoch
                    state.global_step = epoch * 10
                    control = MagicMock()
                    for cb in self._callbacks:
                        cb.on_epoch_end(self._args, state, control)

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=CapturingTrainer),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert len(train_kwargs) == 1, "trainer.train() must be called exactly once"
        assert train_kwargs[0]["resume_from_checkpoint"] == expected_ckpt, (
            f"Expected resume_from_checkpoint={expected_ckpt!r}, "
            f"got {train_kwargs[0]['resume_from_checkpoint']!r}"
        )

    def test_copy_adapter_weights_not_called_for_staging_when_resuming(
        self, tmp_path: Path
    ) -> None:
        """When resuming, copy_adapter_weights(episodic→in_training) must NOT be called.

        The in_training weights are already the mid-training state from the
        prior interrupted run.  Re-copying production weights over them would
        throw away the partial training progress.
        """
        bt = _make_bt(tmp_path)
        job = _make_job()

        self._plant_resume_state(tmp_path, bt, job, epoch=1)

        copy_calls: list[tuple] = []

        def _spy_copy(model, src, dst):
            copy_calls.append((src, dst))

        class _NullTrainer:
            def __init__(self, **kwargs):
                self._callbacks = kwargs.get("callbacks", [])
                self._args = kwargs.get("args", MagicMock(num_train_epochs=3, output_dir="/tmp"))

            def train(self, resume_from_checkpoint=None):
                pass

        with (
            patch("paramem.models.loader.copy_adapter_weights", side_effect=_spy_copy),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_NullTrainer),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        staging_copy = ("episodic", "in_training")
        assert staging_copy not in copy_calls, (
            f"copy_adapter_weights(episodic→in_training) must NOT be called when resuming. "
            f"Got copy calls: {copy_calls}"
        )


# ---------------------------------------------------------------------------
# Resume with mismatching fingerprint
# ---------------------------------------------------------------------------


class TestResumeMismatchedFingerprint:
    """When fingerprints do not match, stale state is wiped and fresh training starts."""

    def _plant_stale_state(self, tmp_path: Path, adapter_name: str = "episodic") -> Path:
        """Write a resume state that does NOT match the test job's fingerprints."""
        checkpoint_dir = tmp_path / "in_training" / "bg_checkpoint" / "checkpoint-50"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "adapter_name": adapter_name,
            "inference_fallback_adapter": "episodic",
            "training_config_fingerprint": "deadbeef" * 8,  # wrong fingerprint
            "keyed_pairs_fingerprint": "cafebabe" * 8,  # wrong fingerprint
            "total_epochs": 10,
            "last_completed_epoch": 5,
            "checkpoint_path": str(checkpoint_dir),
            "started_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        state_path = tmp_path / "in_training" / _RESUME_STATE_FILE
        _write_resume_state_atomic(state_path, state)
        return state_path

    def test_stale_state_wiped_on_fingerprint_mismatch(self, tmp_path: Path) -> None:
        """Stale resume_state.json is removed when fingerprints do not match."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        state_path = self._plant_stale_state(tmp_path)
        assert state_path.exists()

        class _NullTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                pass

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_NullTrainer),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert not state_path.exists(), (
            "Stale resume_state.json must be removed when fingerprints do not match"
        )

    def test_stale_checkpoints_wiped_on_fingerprint_mismatch(self, tmp_path: Path) -> None:
        """Stale bg_checkpoint/ is removed when fingerprints do not match."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        self._plant_stale_state(tmp_path)
        stale_checkpoint_dir = tmp_path / "in_training" / "bg_checkpoint"
        assert stale_checkpoint_dir.exists()

        class _NullTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                pass

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_NullTrainer),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert not stale_checkpoint_dir.exists(), (
            "Stale bg_checkpoint/ must be removed when fingerprints do not match"
        )

    def test_fresh_training_starts_after_mismatch(self, tmp_path: Path) -> None:
        """After wiping stale state, copy_adapter_weights stages from production."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        self._plant_stale_state(tmp_path)

        copy_calls: list[tuple] = []

        def _spy_copy(model, src, dst):
            copy_calls.append((src, dst))

        class _NullTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                pass

        with (
            patch("paramem.models.loader.copy_adapter_weights", side_effect=_spy_copy),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_NullTrainer),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        staging_copy = ("episodic", "in_training")
        assert staging_copy in copy_calls, (
            f"After stale state wipe, fresh training must stage from production. "
            f"Got copy calls: {copy_calls}"
        )

    def test_train_called_without_resume_checkpoint_after_mismatch(self, tmp_path: Path) -> None:
        """trainer.train(resume_from_checkpoint=None) after fingerprint mismatch."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        self._plant_stale_state(tmp_path)

        train_kwargs: list[dict] = []

        class _CapturingTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                train_kwargs.append({"resume_from_checkpoint": resume_from_checkpoint})

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_CapturingTrainer),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert len(train_kwargs) == 1
        assert train_kwargs[0]["resume_from_checkpoint"] is None, (
            f"After fingerprint mismatch, train must be called with resume_from_checkpoint=None. "
            f"Got {train_kwargs[0]['resume_from_checkpoint']!r}"
        )


# ---------------------------------------------------------------------------
# On training exception: resume state preserved
# ---------------------------------------------------------------------------


class TestExceptionPreservesResumeState:
    """When training raises, resume_state.json stays on disk for next restart."""

    def test_resume_state_survives_training_exception(self, tmp_path: Path) -> None:
        """resume_state.json written by a previous epoch is NOT removed on exception."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        state_path = bt._resume_state_path()

        def _spy_write(epoch: int, checkpoint_dir: str) -> None:
            ckpt = Path(checkpoint_dir) / f"checkpoint-{epoch * 10}"
            ckpt.mkdir(parents=True, exist_ok=True)
            # Write a real state so we can check it survives.
            state = {
                "adapter_name": "episodic",
                "last_completed_epoch": epoch,
                "checkpoint_path": str(ckpt),
                "total_epochs": 3,
            }
            _write_resume_state_atomic(state_path, state)

        bt._write_resume_state = _spy_write

        class _FailAfterEpoch1:
            def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
                self._callbacks = callbacks
                self._args = args

            def train(self, resume_from_checkpoint=None):
                # Fire one epoch callback to create the state file.
                state = MagicMock()
                state.epoch = 1
                state.global_step = 10
                control = MagicMock()
                for cb in self._callbacks:
                    cb.on_epoch_end(self._args, state, control)
                # Then fail.
                raise RuntimeError("simulated mid-training crash")

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_FailAfterEpoch1),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            with pytest.raises(RuntimeError, match="simulated mid-training crash"):
                bt._train_adapter(job)

        assert state_path.exists(), (
            "resume_state.json must survive a training exception so the next "
            "restart can resume from the last completed epoch"
        )
        saved = _read_resume_state(state_path)
        assert saved is not None
        assert saved["last_completed_epoch"] == 1

    def test_bg_checkpoint_survives_training_exception(self, tmp_path: Path) -> None:
        """bg_checkpoint/ is NOT wiped when training raises."""
        bt = _make_bt(tmp_path)
        job = _make_job()

        checkpoint_root = tmp_path / "in_training" / "bg_checkpoint"

        def _spy_write(epoch: int, checkpoint_dir: str) -> None:
            ckpt = Path(checkpoint_dir) / f"checkpoint-{epoch * 10}"
            ckpt.mkdir(parents=True, exist_ok=True)

        bt._write_resume_state = _spy_write

        class _FailAfterEpoch1:
            def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
                self._callbacks = callbacks
                self._args = args

            def train(self, resume_from_checkpoint=None):
                state = MagicMock()
                state.epoch = 1
                state.global_step = 10
                control = MagicMock()
                for cb in self._callbacks:
                    cb.on_epoch_end(self._args, state, control)
                raise RuntimeError("crash")

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch("paramem.server.background_trainer.Trainer", new=_FailAfterEpoch1),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            with pytest.raises(RuntimeError, match="crash"):
                bt._train_adapter(job)

        assert checkpoint_root.exists(), (
            "bg_checkpoint/ must NOT be wiped on training exception — "
            "it is needed for the next resume"
        )


# ---------------------------------------------------------------------------
# Successful completion wipes both state file and checkpoint dir
# ---------------------------------------------------------------------------


class TestSuccessfulCompletionCleanup:
    """After a full successful run, both resume artefacts are removed."""

    def test_state_and_checkpoint_both_wiped_on_success(self, tmp_path: Path) -> None:
        bt = _make_bt(tmp_path)
        job = _make_job()

        state_path = bt._resume_state_path()
        checkpoint_root = tmp_path / "in_training" / "bg_checkpoint"

        def _spy_write(epoch: int, checkpoint_dir: str) -> None:
            ckpt = Path(checkpoint_dir) / f"checkpoint-{epoch * 10}"
            ckpt.mkdir(parents=True, exist_ok=True)
            state = {
                "adapter_name": "episodic",
                "last_completed_epoch": epoch,
                "checkpoint_path": str(ckpt),
                "total_epochs": 3,
            }
            _write_resume_state_atomic(state_path, state)

        bt._write_resume_state = _spy_write

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter"),
            patch(
                "paramem.server.background_trainer.format_indexed_training",
                return_value=[{"input_ids": [0], "labels": [0]}],
            ),
            patch(
                "paramem.server.background_trainer.Trainer",
                new=_fake_trainer_class(),
            ),
            patch(
                "paramem.server.background_trainer.TrainingArguments",
                side_effect=_fake_training_arguments,
            ),
        ):
            bt._train_adapter(job)

        assert not state_path.exists(), "resume_state.json must be removed after success"
        assert not checkpoint_root.exists(), "bg_checkpoint/ must be removed after success"


# ---------------------------------------------------------------------------
# Manifest: meta.json written in slot + registry hash reads disk (§2.4.3)
# ---------------------------------------------------------------------------


class TestBgTrainerManifest:
    """_commit_staging_to_production embeds meta.json in the adapter slot."""

    def _configure_model_for_manifest(self, bt):
        """Set JSON-serialisable attributes on the stub model so build_manifest_for works."""
        bt.model.config._name_or_path = "test-base-model"
        bt.model.config._commit_hash = None
        bt.model.base_model.model.state_dict.return_value = {}
        bt.tokenizer.name_or_path = "test-tokenizer"
        bt.tokenizer.backend_tokenizer = None
        bt.tokenizer.vocab_size = 32000
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        bt.model.peft_config["episodic"] = lora_cfg

    def test_bg_trainer_writes_meta_json_in_slot(self, tmp_path: Path) -> None:
        """meta.json must be present in the timestamped slot after commit.

        model.save_pretrained writes stub adapter files so atomic_save_adapter
        can complete the six-step sequence and write meta.json alongside them.
        """
        from paramem.adapters.manifest import AdapterManifest, read_manifest

        bt = _make_bt(tmp_path)
        self._configure_model_for_manifest(bt)

        # model.save_pretrained writes stub files into the pending slot.
        def _fake_save_pretrained(path, selected_adapters=None):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        bt.model.save_pretrained.side_effect = _fake_save_pretrained

        with patch("paramem.models.loader.copy_adapter_weights"):
            bt._commit_staging_to_production("episodic")

        adapter_dir = tmp_path / "episodic"
        slots = [d for d in adapter_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, f"No slot dir created under {adapter_dir}"
        slot = slots[0]

        assert (slot / "meta.json").exists(), f"meta.json missing from slot {slot}"

        manifest = read_manifest(slot)
        assert isinstance(manifest, AdapterManifest)
        assert manifest.name == "episodic"

    def test_bg_trainer_manifest_registry_hash_reads_disk(self, tmp_path: Path) -> None:
        """manifest.registry_sha256 must equal sha256 of the on-disk registry.

        BackgroundTrainer never calls save_bytes (§2.4.3): it is read-only on
        the registry.  The manifest's registry_sha256 is derived by reading the
        on-disk file directly inside build_manifest_for.
        """
        import hashlib

        from paramem.adapters.manifest import read_manifest
        from paramem.training.key_registry import KeyRegistry

        # Seed an on-disk registry file.
        reg = KeyRegistry()
        reg.add("graph1", adapter_id="episodic")
        registry_path = tmp_path / "indexed_key_registry.json"
        reg.save(registry_path)
        expected_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()

        bt = _make_bt(tmp_path)
        self._configure_model_for_manifest(bt)

        def _fake_save_pretrained(path, selected_adapters=None):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        bt.model.save_pretrained.side_effect = _fake_save_pretrained

        save_bytes_called = []

        with (
            patch("paramem.models.loader.copy_adapter_weights"),
            patch.object(
                KeyRegistry,
                "save_bytes",
                side_effect=lambda: save_bytes_called.append(1) or reg.save_bytes(),
            ),
        ):
            bt._commit_staging_to_production("episodic")

        assert save_bytes_called == [], (
            "BackgroundTrainer must NOT call save_bytes — registry is read-only (§2.4.3)"
        )

        adapter_dir = tmp_path / "episodic"
        slots = [d for d in adapter_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, "No slot created"
        manifest = read_manifest(slots[0])
        assert manifest.registry_sha256 == expected_hash, (
            f"Expected registry_sha256={expected_hash!r}, got {manifest.registry_sha256!r}"
        )
