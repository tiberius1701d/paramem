"""Unit tests for the budget/donor validation-arm additions to
``experiments/test20_smallN_cold_gate.py`` (``--lr-decay-steps``,
``--donor-init``, ``--donor-checkpoint``).

CPU-only, no GPU / real model weights. The functions under test are pure
Python logic (label derivation), filesystem/marker resolution
(``_resolve_donor_source``, ``_read_donor_meta``), and CLI parsing. The
GPU-touching training path (``_build_donor_checkpoint``) is monkeypatched
out exactly as ``tests/test_donor.py`` mocks PEFT/training primitives for
``paramem.training.donor.build_donor`` — this file follows that same
project convention for its one production-adjacent (but experiment-owned)
counterpart.

New file justification: no prior test file exercises
``experiments/test20_smallN_cold_gate.py``'s internal helpers directly (only
the structural import-boundary guard, ``tests/test_experiment_boundary.py``,
touches it) — this is the first, scoped to exactly the functions this
change added or changed the signature of.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experiments.test20_smallN_cold_gate import (
    DONOR_BUILD_MARKER_FILENAME,
    DONOR_META_FILENAME,
    _build_donor_checkpoint,
    _condition_label,
    _default_arm_label,
    _parse_args,
    _read_donor_meta,
    _resolve_donor_source,
)


def _write_slot(tmp_path: Path, name: str, weights: bytes = b"fake-weights") -> Path:
    """Create a donor checkpoint slot with real weights + a matching, valid
    ``donor_meta.json`` — the shape both ``_read_donor_meta`` and every
    ``_resolve_donor_source`` reuse branch require."""
    slot = tmp_path / name
    slot.mkdir()
    (slot / "adapter_model.safetensors").write_bytes(weights)
    meta = {
        "seed": 42,
        "n_entries": 128,
        "epochs": 30,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
    }
    (slot / DONOR_META_FILENAME).write_text(json.dumps(meta))
    return slot


class TestConditionLabel:
    """``_condition_label`` — descriptive, never letter-labeled condition names."""

    def test_cold(self):
        assert _condition_label("cold", 50) == "cold 50ep"

    def test_donor(self):
        assert _condition_label("donor", 30) == "donor-init 30ep"

    def test_warm(self):
        assert _condition_label("warm", 30) == "warm-from-adapter 30ep"

    def test_unknown_mode_raises(self):
        with pytest.raises(KeyError):
            _condition_label("bogus", 30)


class TestDefaultArmLabel:
    """``_default_arm_label`` — mode-string signature (was a ``warm: bool``)."""

    def test_synthetic_cold_label_unchanged(self):
        """Byte-identical to the pre-change ``warm=False`` output — --resume
        must keep finding runs launched before --donor-init existed."""
        assert _default_arm_label(3, 60, is_real=False, mode="cold") == "cold_n3_s60"

    def test_synthetic_warm_label_unchanged(self):
        """Byte-identical to the pre-change ``warm=True`` output."""
        assert _default_arm_label(3, 60, is_real=False, mode="warm") == "n3_warm_s60"

    def test_real_cold_label_unchanged(self):
        assert _default_arm_label(3, 60, is_real=True, mode="cold") == "real3_cold_s60"

    def test_real_warm_label_unchanged(self):
        assert _default_arm_label(3, 60, is_real=True, mode="warm") == "real3_warm_s60"

    def test_synthetic_donor_label_is_new_and_distinct(self):
        label = _default_arm_label(12, 180, is_real=False, mode="donor")
        assert label == "n12_donor_s180"
        assert label != _default_arm_label(12, 180, is_real=False, mode="cold")
        assert label != _default_arm_label(12, 180, is_real=False, mode="warm")

    def test_real_donor_label_is_new_and_distinct(self):
        label = _default_arm_label(21, 550, is_real=True, mode="donor")
        assert label == "real21_donor_s550"
        assert label != _default_arm_label(21, 550, is_real=True, mode="cold")
        assert label != _default_arm_label(21, 550, is_real=True, mode="warm")


class TestReadDonorMeta:
    """``_read_donor_meta`` — M4: SHA-256 verification against the recorded value."""

    def test_valid_slot_returns_meta(self, tmp_path):
        slot = _write_slot(tmp_path, "valid_slot", weights=b"real-donor-weights")
        meta = _read_donor_meta(slot)
        assert meta["seed"] == 42
        assert meta["n_entries"] == 128
        assert meta["epochs"] == 30
        assert meta["weights_sha256"] == hashlib.sha256(b"real-donor-weights").hexdigest()

    def test_missing_meta_file_raises(self, tmp_path):
        slot = tmp_path / "no_meta"
        slot.mkdir()
        (slot / "adapter_model.safetensors").write_bytes(b"weights")
        with pytest.raises(SystemExit, match=DONOR_META_FILENAME):
            _read_donor_meta(slot)

    def test_sha_mismatch_raises(self, tmp_path):
        slot = _write_slot(tmp_path, "tampered_slot", weights=b"original-weights")
        # Simulate post-build corruption/modification.
        (slot / "adapter_model.safetensors").write_bytes(b"tampered-weights")
        with pytest.raises(SystemExit, match="mismatch"):
            _read_donor_meta(slot)


class TestResolveDonorSource:
    """``_resolve_donor_source`` — the three resolution branches.

    Return signature is ``(model, slot, built_fresh, donor_meta)`` — B2
    (cooldown-after-build) reads ``built_fresh``; H1 (key-overlap recording)
    reads ``donor_meta``.
    """

    def test_external_checkpoint_reused_without_building(self, tmp_path, monkeypatch):
        slot = _write_slot(tmp_path, "external_slot")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        build_mock = MagicMock()
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._build_donor_checkpoint", build_mock
        )

        model = object()
        result_model, result_slot, built_fresh, donor_meta = _resolve_donor_source(
            str(slot), run_dir, model, MagicMock(), MagicMock(), MagicMock()
        )

        assert result_model is model
        assert result_slot == slot
        assert built_fresh is False
        assert donor_meta["seed"] == 42
        build_mock.assert_not_called()
        assert not (run_dir / DONOR_BUILD_MARKER_FILENAME).exists()

    def test_external_checkpoint_missing_weights_raises(self, tmp_path):
        empty_dir = tmp_path / "no_weights_here"
        empty_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with pytest.raises(SystemExit, match="adapter_model.safetensors"):
            _resolve_donor_source(
                str(empty_dir), run_dir, object(), MagicMock(), MagicMock(), MagicMock()
            )

    def test_external_checkpoint_sha_mismatch_fails_loud(self, tmp_path):
        """M4: --donor-checkpoint reuse must verify the checkpoint SHA against
        its own donor_meta.json — a tampered/corrupted slot fails loud rather
        than silently seeding from unverified weights."""
        slot = _write_slot(tmp_path, "tampered_external_slot", weights=b"original")
        (slot / "adapter_model.safetensors").write_bytes(b"tampered")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with pytest.raises(SystemExit, match="mismatch"):
            _resolve_donor_source(
                str(slot), run_dir, object(), MagicMock(), MagicMock(), MagicMock()
            )

    def test_reuses_existing_build_marker_without_rebuilding(self, tmp_path, monkeypatch):
        """A crash AFTER the donor build (marker written) must not rebuild it
        on retry — the resumability requirement. The marker itself is now
        minimal (slot + timestamp only); provenance lives in the slot's own
        donor_meta.json."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        slot = _write_slot(tmp_path, "already_built_slot")
        marker = {"slot": str(slot), "timestamp": 1234567890}
        (run_dir / DONOR_BUILD_MARKER_FILENAME).write_text(json.dumps(marker))

        build_mock = MagicMock()
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._build_donor_checkpoint", build_mock
        )

        model = object()
        result_model, result_slot, built_fresh, donor_meta = _resolve_donor_source(
            None, run_dir, model, MagicMock(), MagicMock(), MagicMock()
        )

        assert result_model is model
        assert result_slot == slot
        assert built_fresh is False
        assert donor_meta["seed"] == 42
        build_mock.assert_not_called()

    def test_marker_pointing_at_missing_checkpoint_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        marker = {"slot": str(tmp_path / "vanished_slot"), "timestamp": 1234567890}
        (run_dir / DONOR_BUILD_MARKER_FILENAME).write_text(json.dumps(marker))

        with pytest.raises(SystemExit, match="Donor build marker"):
            _resolve_donor_source(None, run_dir, object(), MagicMock(), MagicMock(), MagicMock())

    def test_builds_fresh_and_writes_marker_when_nothing_exists(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        slot = _write_slot(tmp_path, "freshly_built_slot")
        donor_summary = {
            "seed": 42,
            "n_entries": 128,
            "epochs": 30,
            "weights_sha256": "cafef00d",
        }
        built_model = object()
        build_mock = MagicMock(return_value=(built_model, slot, donor_summary))
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._build_donor_checkpoint", build_mock
        )

        input_model = object()
        tokenizer = MagicMock()
        adapter_config = MagicMock()
        training_config = MagicMock()
        result_model, result_slot, built_fresh, donor_meta = _resolve_donor_source(
            None, run_dir, input_model, tokenizer, adapter_config, training_config
        )

        assert result_model is built_model
        assert result_slot == slot
        assert built_fresh is True  # B2: caller must cool down before the first seed
        assert donor_meta == {
            "seed": 42,
            "n_entries": 128,
            "epochs": 30,
            "weights_sha256": "cafef00d",
        }
        build_mock.assert_called_once_with(
            input_model,
            tokenizer,
            adapter_config,
            training_config,
            run_dir / "donor_checkpoint",
        )

        marker_path = run_dir / DONOR_BUILD_MARKER_FILENAME
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text())
        assert marker == {"slot": str(slot), "timestamp": marker["timestamp"]}

    def test_second_call_after_fresh_build_reuses_marker(self, tmp_path, monkeypatch):
        """End-to-end resumability: build once, then a second call (simulating
        a resumed process) must not invoke the builder again."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        slot = _write_slot(tmp_path, "freshly_built_slot")
        donor_summary = {
            "seed": 42,
            "n_entries": 128,
            "epochs": 30,
            "weights_sha256": hashlib.sha256(b"fake-weights").hexdigest(),
        }
        build_mock = MagicMock(return_value=(object(), slot, donor_summary))
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._build_donor_checkpoint", build_mock
        )

        _, _, built_fresh_1, _ = _resolve_donor_source(
            None, run_dir, object(), MagicMock(), MagicMock(), MagicMock()
        )
        assert build_mock.call_count == 1
        assert built_fresh_1 is True

        _, _, built_fresh_2, _ = _resolve_donor_source(
            None, run_dir, object(), MagicMock(), MagicMock(), MagicMock()
        )
        assert build_mock.call_count == 1  # not called again
        assert built_fresh_2 is False


class TestLrDecayStepsCli:
    """``--lr-decay-steps`` CLI parsing (trivial coverage for the docstring's claim)."""

    def test_default_is_none(self):
        import sys

        argv = sys.argv
        sys.argv = ["test20"]
        try:
            args = _parse_args()
        finally:
            sys.argv = argv
        assert args.lr_decay_steps is None

    def test_explicit_value_threads_through(self):
        import sys

        argv = sys.argv
        sys.argv = ["test20", "--lr-decay-steps", "550"]
        try:
            args = _parse_args()
        finally:
            sys.argv = argv
        assert args.lr_decay_steps == 550


class TestBuildDonorCheckpointPinsLrDecaySteps:
    """M1: the donor's OWN training must never inherit the arm's
    --lr-decay-steps override — it always trains at lr_decay_steps=None
    (the anchored-bucket recipe), regardless of *base_training_config*.
    """

    def test_donor_training_config_pins_lr_decay_steps_none(self, tmp_path, monkeypatch):
        import dataclasses

        from paramem.utils.config import TrainingConfig

        donor_entries_mock = MagicMock(
            return_value=[
                {"key": f"graph{i}", "subject": "s", "predicate": "p", "object": f"o{i}"}
                for i in range(3)
            ]
        )
        monkeypatch.setattr("experiments.test20_smallN_cold_gate.donor_entries", donor_entries_mock)
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.build_registry", MagicMock(return_value={})
        )
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.create_adapter",
            MagicMock(side_effect=lambda model, cfg, name: model),
        )
        monkeypatch.setattr("experiments.test20_smallN_cold_gate.switch_adapter", MagicMock())
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.lora_b_frobenius_norm",
            MagicMock(side_effect=[0.0, 1.23]),
        )
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.format_entry_training",
            MagicMock(return_value=[{"input_ids": [1]}, {"input_ids": [2]}, {"input_ids": [3]}]),
        )

        captured_training_config = {}

        def _fake_train_adapter(**kwargs):
            captured_training_config["value"] = kwargs["training_config"]
            return {"train_loss": 0.01}

        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.train_adapter",
            MagicMock(side_effect=_fake_train_adapter),
        )
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._run_recall_probe",
            MagicMock(
                return_value={
                    "exact_count": 3,
                    "total": 3,
                    "rate": 1.0,
                    "mean_confidence": 1.0,
                    "per_key": [],
                }
            ),
        )

        def _fake_atomic_save_adapter(model, checkpoint_root, adapter_name):
            slot = checkpoint_root / "20260726-000000"
            slot.mkdir(parents=True)
            (slot / "adapter_model.safetensors").write_bytes(b"donor-weights")
            return slot

        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.atomic_save_adapter",
            MagicMock(side_effect=_fake_atomic_save_adapter),
        )

        # The arm's own --lr-decay-steps override -- must NOT leak into the
        # donor's training config.
        base_training_config = dataclasses.replace(
            TrainingConfig(), lr_decay_steps=550, max_seq_length=1024
        )
        checkpoint_root = tmp_path / "donor_checkpoint"

        model, slot, donor_summary = _build_donor_checkpoint(
            object(), MagicMock(), MagicMock(), base_training_config, checkpoint_root
        )

        assert captured_training_config["value"].lr_decay_steps is None
        assert (slot / DONOR_META_FILENAME).is_file()
        meta = json.loads((slot / DONOR_META_FILENAME).read_text())
        assert meta["weights_sha256"] == donor_summary["weights_sha256"]
