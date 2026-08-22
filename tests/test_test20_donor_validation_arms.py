"""Unit tests for the budget/donor validation-arm additions to
``experiments/test20_smallN_cold_gate.py`` (``--lr-decay-steps``,
``--accum``, ``--donor-init``, ``--donor-checkpoint``).

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

``_read_donor_meta``/``_resolve_donor_source`` validation against a real
manifest-shaped slot, and ``build_donor``'s own save/manifest-write path,
are covered in ``tests/test_donor.py`` (``TestResolveDonorCheckpoint``,
``TestDonorCheckpointValidityAcrossKeyLifecycleEvents``), not here — this
file stays scoped to the CLI/label/marker-resolution helpers named above.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from experiments.test20_smallN_cold_gate import (
    DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME,
    _condition_label,
    _default_arm_label,
    _expected_optimizer_steps,
    _parse_args,
    _run_donor_build_smoke,
    _steps_per_epoch,
    main,
)
from paramem.server.config import load_server_config
from paramem.utils.config import budget_for

_BASE_ID = "test/base-model"
_LORA_SHAPE = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}
# (base_model_id, lora_shape) — what _read_donor_meta verifies a slot against.
_VERIFY = (_BASE_ID, _LORA_SHAPE)


def _stub_model():
    """Model stub whose ``_donor_verification_context`` yields ``_BASE_ID``."""
    return SimpleNamespace(config=SimpleNamespace(_name_or_path=_BASE_ID))


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


class TestAccumCli:
    """``--accum`` CLI parsing — default None preserves today's behaviour;
    an explicit value threads through to ``args.accum`` unchanged."""

    def test_default_is_none(self):
        import sys

        argv = sys.argv
        sys.argv = ["test20"]
        try:
            args = _parse_args()
        finally:
            sys.argv = argv
        assert args.accum is None

    def test_explicit_value_threads_through(self):
        import sys

        argv = sys.argv
        sys.argv = ["test20", "--accum", "1"]
        try:
            args = _parse_args()
        finally:
            sys.argv = argv
        assert args.accum == 1


class TestExpectedOptimizerStepsDerivation:
    """``_expected_optimizer_steps`` derives from the SAME resolved values
    the run actually trains with — never a hardcoded module
    constant. These are real parity checks against
    ``paramem.utils.config.budget_for`` and the loaded fixture, not a
    self-comparison (the prior ``test_default_accum_matches_recipe_value``
    compared ``_RECIPE_GRAD_ACCUM_STEPS`` against itself and could never
    fail — this replaces it)."""

    def test_matches_budget_for_at_the_donor_population_size(self):
        """budget_for(147) — the donor's own population size
        (DONOR_MIN_ENTRIES=128 rounds up to 147) — is the actual derivation
        _build_donor_checkpoint uses; assert against budget_for's real
        output, not a value copied into a module constant that could
        silently drift from paramem.utils.config."""
        epochs, accum, lr_decay_steps = budget_for(147)
        assert (epochs, accum, lr_decay_steps) == (30, 2, None)
        assert _expected_optimizer_steps(147, epochs, accum, batch_size=1) == (
            _steps_per_epoch(147, 1, accum) * epochs
        )

    def test_matches_budget_for_at_the_lt16_bucket(self):
        """N=3 falls in the ``<16`` bucket (accum=1, 80 epochs) — a
        DIFFERENT bucket than N=147's, proving the derivation is genuinely
        per-N rather than a single hardcoded pair."""
        epochs, accum, lr_decay_steps = budget_for(3)
        assert (epochs, accum, lr_decay_steps) == (80, 1, None)
        assert _expected_optimizer_steps(3, epochs, accum, batch_size=1) == (
            _steps_per_epoch(3, 1, accum) * epochs
        )

    def test_fixture_batch_size_matches_loaded_training_config(self):
        """The fixture-sourced field this harness treats as ground truth
        (batch_size) actually matches ``tests/fixtures/server.yaml`` —
        catches silent fixture drift."""
        cfg = load_server_config("tests/fixtures/server.yaml")
        assert cfg.training_config.batch_size == 1

    def test_explicit_accum_changes_the_result(self):
        """A caller-supplied accum (e.g. ``--accum``) must actually change
        the derived step count — proves the function has no internal
        fallback to a hardcoded default."""
        assert _expected_optimizer_steps(3, 80, accum=1, batch_size=1) != (
            _expected_optimizer_steps(3, 80, accum=2, batch_size=1)
        )


class TestDonorBuildSmokeConflictingFlagsDerivedFromParser:
    """``main()``'s ``--donor-build-smoke`` flag-conflict guard derives
    "is this flag set" from the CLI parser's own defaults
    (``vars(args)`` vs a ``parse_args([])`` baseline of the SAME parser)
    rather than a hand-maintained mirror of the flag list — every dest
    except the allowed trio (``--model``/``--resume``/``--donor-build-smoke``
    itself) is a conflicting flag, so a newly-added flag is covered
    automatically instead of silently bypassing the guard."""

    @pytest.mark.parametrize(
        "flag_args",
        [
            ["--n-entries", "3"],
            ["--entries-json", "some_file.json"],
            ["--epochs", "10"],
            ["--warm-from", "/some/donor/dir"],
            ["--arm", "custom_arm"],
            ["--seeds", "42"],
            ["--probe-before-training"],
            ["--lr-decay-steps", "100"],
            ["--accum", "2"],
            ["--donor-init"],
            ["--donor-checkpoint", "/some/slot"],
        ],
        ids=[
            "n-entries",
            "entries-json",
            "epochs",
            "warm-from",
            "arm",
            "seeds",
            "probe-before-training",
            "lr-decay-steps",
            "accum",
            "donor-init",
            "donor-checkpoint",
        ],
    )
    def test_each_conflicting_flag_fails_loud(self, monkeypatch, flag_args):
        monkeypatch.setattr(sys, "argv", ["test20", "--donor-build-smoke", *flag_args])
        with pytest.raises(SystemExit, match="mutually exclusive"):
            main()

    def test_model_and_resume_are_not_conflicting(self, monkeypatch):
        """``--model``/``--resume`` are the allowed trio (alongside
        ``--donor-build-smoke`` itself) — this must reach
        ``_main_donor_build_smoke`` rather than raising the conflict
        SystemExit (mocked out here since it would otherwise touch the GPU)."""
        monkeypatch.setattr(sys, "argv", ["test20", "--donor-build-smoke", "--model", "mistral"])
        main_smoke_mock = MagicMock()
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._main_donor_build_smoke", main_smoke_mock
        )
        main()
        main_smoke_mock.assert_called_once()


@pytest.fixture
def _mock_cuda_telemetry(monkeypatch):
    """CPU-safe stand-ins for the CUDA telemetry calls
    ``_run_donor_build_smoke`` makes unconditionally in its build phase, so
    these unit tests never touch a real CUDA context regardless of whether
    the test host has a GPU."""
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", MagicMock())
    monkeypatch.setattr("torch.cuda.max_memory_allocated", MagicMock(return_value=0.0))
    monkeypatch.setattr("torch.cuda.max_memory_reserved", MagicMock(return_value=0.0))
    monkeypatch.setattr(
        "experiments.test20_smallN_cold_gate._cuda_mem_get_info_mib",
        MagicMock(return_value={"free_mib": 1.0, "total_mib": 2.0}),
    )


class TestRunDonorBuildSmokeTwoMarkerResume:
    """``_run_donor_build_smoke``'s own two resumability markers — SEPARATE
    from ``_build_or_reuse_own_donor_checkpoint``'s own
    ``DONOR_BUILD_MARKER_FILENAME`` check (already covered by
    ``TestResolveDonorSource``): ``build_results.json`` (build-phase result
    skip) and ``DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME`` (seed-phase
    no-op)."""

    def test_existing_build_results_are_not_recomputed(
        self, tmp_path, monkeypatch, _mock_cuda_telemetry
    ):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "build_results.json").write_text(json.dumps({"sentinel": "pre-existing"}))
        # Seed marker ALSO present so phase 2 is a no-op without further
        # mocking -- this test isolates the build-results skip.
        (run_dir / DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME).write_text(
            json.dumps({"timestamp": 1, "success": True})
        )

        slot = tmp_path / "slot"
        slot.mkdir()
        build_mock = MagicMock(
            return_value=(_stub_model(), slot, False, {"seed": 42, "n_entries": 147, "epochs": 30})
        )
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._build_or_reuse_own_donor_checkpoint", build_mock
        )
        read_meta_mock = MagicMock()
        monkeypatch.setattr("experiments.test20_smallN_cold_gate._read_donor_meta", read_meta_mock)

        cfg = load_server_config("tests/fixtures/server.yaml")
        result = _run_donor_build_smoke(
            _stub_model(),
            MagicMock(),
            cfg,
            run_dir,
            {"free_mib": 1.0, "total_mib": 2.0},
            {"free_mib": 1.0, "total_mib": 2.0},
        )

        assert result is None
        build_mock.assert_called_once()
        # The build-phase results-recording branch (which reads
        # donor_meta.json via _read_donor_meta) never ran.
        read_meta_mock.assert_not_called()
        # build_results.json is untouched -- proves the build-phase results
        # were not recomputed/overwritten.
        assert json.loads((run_dir / "build_results.json").read_text()) == {
            "sentinel": "pre-existing"
        }

    def test_existing_seed_marker_makes_seed_phase_a_no_op(
        self, tmp_path, monkeypatch, _mock_cuda_telemetry
    ):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME).write_text(
            json.dumps({"timestamp": 1, "success": True})
        )

        slot = tmp_path / "slot"
        slot.mkdir()
        (slot / "adapter_model.safetensors").write_bytes(b"weights")
        build_mock = MagicMock(
            return_value=(_stub_model(), slot, False, {"seed": 42, "n_entries": 147, "epochs": 30})
        )
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate._build_or_reuse_own_donor_checkpoint", build_mock
        )
        read_meta_mock = MagicMock(
            return_value={
                "seed": 42,
                "n_entries": 147,
                "epochs": 30,
                "gradient_accumulation_steps": 2,
                "realized_optimizer_steps": 2220,
                # _read_donor_meta itself now sources this from
                # read_manifest(slot).payload.sha256 (never a
                # donor_meta.json field) -- _read_donor_meta is mocked
                # wholesale here, so only the returned dict's SHAPE matters.
                "weights_sha256": "a" * 64,
                "wall_train_seconds": 456.7,
            }
        )
        monkeypatch.setattr("experiments.test20_smallN_cold_gate._read_donor_meta", read_meta_mock)
        # Defensive mocks for the seed-phase collaborators -- must NEVER be
        # invoked once the seed marker short-circuits the function; kept
        # here so a regression that moves the marker check surfaces as a
        # clean assertion failure rather than a real adapter-load attempt.
        # The seed phase mounts the resolved donor slot via mount_adapter
        # (paramem.models.loader) -- PeftModel.from_pretrained is no longer
        # on this script's load path.
        mount_adapter_mock = MagicMock()
        monkeypatch.setattr("experiments.test20_smallN_cold_gate.mount_adapter", mount_adapter_mock)
        create_adapter_mock = MagicMock()
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.create_adapter", create_adapter_mock
        )
        copy_weights_mock = MagicMock()
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.copy_adapter_weights", copy_weights_mock
        )

        cfg = load_server_config("tests/fixtures/server.yaml")
        result = _run_donor_build_smoke(
            _stub_model(),
            MagicMock(),
            cfg,
            run_dir,
            {"free_mib": 1.0, "total_mib": 2.0},
            {"free_mib": 1.0, "total_mib": 2.0},
        )

        assert result is None
        build_mock.assert_called_once()
        create_adapter_mock.assert_not_called()
        mount_adapter_mock.assert_not_called()
        copy_weights_mock.assert_not_called()
        # The build-phase results ARE recorded (build_results.json did not
        # pre-exist), proving only the SEED phase was skipped.
        assert (run_dir / "build_results.json").is_file()
