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
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from experiments.test20_smallN_cold_gate import (
    DONOR_BUILD_MARKER_FILENAME,
    DONOR_BUILD_PROVENANCE_FILENAME,
    DONOR_BUILD_SMOKE_SEED_MARKER_FILENAME,
    DONOR_META_FILENAME,
    _build_donor_checkpoint,
    _condition_label,
    _default_arm_label,
    _expected_optimizer_steps,
    _parse_args,
    _read_donor_meta,
    _resolve_donor_source,
    _run_donor_build_smoke,
    _steps_per_epoch,
    main,
)
from paramem.server.config import load_server_config
from paramem.utils.config import AdapterConfig, TrainingConfig, budget_for

_BASE_ID = "test/base-model"
_LORA_SHAPE = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}
# (base_model_id, lora_shape) — what _read_donor_meta verifies a slot against.
_VERIFY = (_BASE_ID, _LORA_SHAPE)


def _write_slot(
    tmp_path: Path,
    name: str,
    weights: bytes = b"fake-weights",
    *,
    provenance: bool = True,
) -> Path:
    """Create a slot that satisfies the PRODUCTION donor contract.

    ``_read_donor_meta`` now verifies through
    ``paramem.training.donor.donor_slot_valid`` — the same rule the production
    seeding hook applies — so a fixture slot needs what any donor slot needs:
    a ``meta.json`` manifest carrying the base model and LoRA shape, and a
    ``donor_meta.json`` whose recorded triple set regenerates from its seed
    and whose digest matches the weights on disk.

    *provenance* writes this script's own build-provenance file alongside;
    ``False`` produces the shape a PRODUCTION-built donor has (valid, but with
    no build fields to report).
    """
    from paramem.adapters.manifest import (
        MANIFEST_SCHEMA_VERSION,
        AdapterManifest,
        BaseModelFingerprint,
        LoRAShape,
        TokenizerFingerprint,
        write_manifest,
    )
    from paramem.training.donor import (
        DONOR_MIN_ENTRIES,
        DONOR_RECIPE_ID,
        donor_entries,
        triples_hash,
    )

    slot = tmp_path / name
    slot.mkdir()
    (slot / "adapter_model.safetensors").write_bytes(weights)
    write_manifest(
        slot,
        AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name=name,
            trained_at="2026-01-01T00:00:00Z",
            base_model=BaseModelFingerprint(repo=_BASE_ID, sha="abc", hash="sha256:def"),
            tokenizer=TokenizerFingerprint(
                name_or_path=_BASE_ID, vocab_size=32768, merges_hash="feed"
            ),
            lora=LoRAShape(
                rank=_LORA_SHAPE["r"],
                alpha=_LORA_SHAPE["lora_alpha"],
                dropout=0.0,
                target_modules=tuple(_LORA_SHAPE["target_modules"]),
            ),
            registry_sha256="",
            key_count=0,
            synthesized=False,
            window_stamp="",
        ),
    )
    entries = donor_entries(42, DONOR_MIN_ENTRIES)
    (slot / DONOR_META_FILENAME).write_text(
        json.dumps(
            {
                "seed": 42,
                "recipe": DONOR_RECIPE_ID,
                "n_requested": DONOR_MIN_ENTRIES,
                "triples": entries,
                "triples_hash": triples_hash(entries),
                "weights_sha256": hashlib.sha256(weights).hexdigest(),
            }
        )
    )
    if provenance:
        (slot / DONOR_BUILD_PROVENANCE_FILENAME).write_text(
            json.dumps(
                {
                    "n_entries": len(entries),
                    "epochs": 30,
                    "gradient_accumulation_steps": 2,
                    "realized_optimizer_steps": 2220,
                    "wall_train_seconds": 123.4,
                }
            )
        )
    return slot


def _stub_model():
    """Model stub whose ``_donor_verification_context`` yields ``_BASE_ID``."""
    return SimpleNamespace(config=SimpleNamespace(_name_or_path=_BASE_ID))


def _stub_adapter_config() -> AdapterConfig:
    """AdapterConfig whose ``lora_shape_fields`` yields ``_LORA_SHAPE``."""
    return AdapterConfig(
        rank=_LORA_SHAPE["r"],
        alpha=_LORA_SHAPE["lora_alpha"],
        target_modules=list(_LORA_SHAPE["target_modules"]),
    )


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
    """``_read_donor_meta`` — SHA-256 verification against the recorded value."""

    def test_valid_slot_returns_meta(self, tmp_path):
        slot = _write_slot(tmp_path, "valid_slot", weights=b"real-donor-weights")
        meta = _read_donor_meta(slot, *_VERIFY)
        assert meta["seed"] == 42
        assert meta["n_entries"] == 147
        assert meta["epochs"] == 30
        assert meta["weights_sha256"] == hashlib.sha256(b"real-donor-weights").hexdigest()

    def test_missing_meta_file_raises(self, tmp_path):
        slot = tmp_path / "no_meta"
        slot.mkdir()
        (slot / "adapter_model.safetensors").write_bytes(b"weights")
        with pytest.raises(SystemExit, match=DONOR_META_FILENAME):
            _read_donor_meta(slot, *_VERIFY)

    def test_sha_mismatch_raises(self, tmp_path):
        slot = _write_slot(tmp_path, "tampered_slot", weights=b"original-weights")
        # Simulate post-build corruption/modification.
        (slot / "adapter_model.safetensors").write_bytes(b"tampered-weights")
        with pytest.raises(SystemExit, match="digest that no longer matches"):
            _read_donor_meta(slot, *_VERIFY)


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

        model = _stub_model()
        result_model, result_slot, built_fresh, donor_meta = _resolve_donor_source(
            str(slot), run_dir, model, MagicMock(), _stub_adapter_config(), MagicMock()
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
                str(empty_dir),
                run_dir,
                _stub_model(),
                MagicMock(),
                _stub_adapter_config(),
                MagicMock(),
            )

    def test_external_checkpoint_sha_mismatch_fails_loud(self, tmp_path):
        """--donor-checkpoint reuse must verify the checkpoint SHA against
        its own donor_meta.json — a tampered/corrupted slot fails loud rather
        than silently seeding from unverified weights."""
        slot = _write_slot(tmp_path, "tampered_external_slot", weights=b"original")
        (slot / "adapter_model.safetensors").write_bytes(b"tampered")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with pytest.raises(SystemExit, match="digest that no longer matches"):
            _resolve_donor_source(
                str(slot), run_dir, _stub_model(), MagicMock(), _stub_adapter_config(), MagicMock()
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

        model = _stub_model()
        result_model, result_slot, built_fresh, donor_meta = _resolve_donor_source(
            None, run_dir, model, MagicMock(), _stub_adapter_config(), MagicMock()
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
            _resolve_donor_source(
                None, run_dir, _stub_model(), MagicMock(), _stub_adapter_config(), MagicMock()
            )

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

        input_model = _stub_model()
        tokenizer = MagicMock()
        adapter_config = _stub_adapter_config()
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
            None, run_dir, _stub_model(), MagicMock(), _stub_adapter_config(), MagicMock()
        )
        assert build_mock.call_count == 1
        assert built_fresh_1 is True

        _, _, built_fresh_2, _ = _resolve_donor_source(
            None, run_dir, _stub_model(), MagicMock(), _stub_adapter_config(), MagicMock()
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


@pytest.fixture
def _donor_build_env(tmp_path, monkeypatch):
    """Shared monkeypatch scaffolding for ``_build_donor_checkpoint`` unit
    tests (collapses the ~70-line duplicated setup previously
    repeated per pinning test into one fixture).

    Patches every collaborator ``_build_donor_checkpoint`` calls
    (``donor_entries``, ``build_registry``, ``create_adapter``,
    ``switch_adapter``, ``lora_b_frobenius_norm``, ``format_entry_training``,
    ``train_adapter``, ``_run_recall_probe``, ``atomic_save_adapter``). The
    fake ``train_adapter`` also fires the realized-step count onto the
    ``_StepCaptureCallback`` passed via ``callbacks_extra`` — mirroring HF
    Trainer's ``on_train_end`` — so ``_build_donor_checkpoint``'s own
    realized-steps assertion sees a populated, DERIVED-correct
    value instead of failing with "callback never fired".

    Returns a ``SimpleNamespace`` with ``.entries`` (the 3-entry donor
    population the mocks use), ``.captured`` (dict populated with
    ``"training_config"`` after the call), and ``.checkpoint_root``.
    """
    entries = [
        {"key": f"graph{i}", "subject": "s", "predicate": "p", "object": f"o{i}"} for i in range(3)
    ]
    monkeypatch.setattr(
        "experiments.test20_smallN_cold_gate.donor_entries", MagicMock(return_value=entries)
    )
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

    captured: dict = {}

    def _fake_train_adapter(**kwargs):
        training_config = kwargs["training_config"]
        captured["training_config"] = training_config
        realized = (
            _steps_per_epoch(
                len(kwargs["train_dataset"]),
                training_config.batch_size,
                training_config.gradient_accumulation_steps,
            )
            * training_config.num_epochs
        )
        for callback in kwargs.get("callbacks_extra", []):
            callback.global_step = realized
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

    def _fake_atomic_save_adapter(model, checkpoint_root, adapter_name, *, manifest=None):
        from paramem.adapters.manifest import write_manifest

        slot = checkpoint_root / "20260726-000000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"donor-weights")
        if manifest is not None:
            write_manifest(slot, manifest)
        return slot

    def _fake_build_manifest_for(model, tokenizer, adapter_name, **kwargs):
        from paramem.adapters.manifest import (
            MANIFEST_SCHEMA_VERSION,
            AdapterManifest,
            BaseModelFingerprint,
            LoRAShape,
            TokenizerFingerprint,
        )

        return AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name=adapter_name,
            trained_at="2026-07-26T00:00:00Z",
            base_model=BaseModelFingerprint(repo=_BASE_ID, sha="abc", hash="sha256:def"),
            tokenizer=TokenizerFingerprint(
                name_or_path=_BASE_ID, vocab_size=32768, merges_hash="feed"
            ),
            lora=LoRAShape(
                rank=_LORA_SHAPE["r"],
                alpha=_LORA_SHAPE["lora_alpha"],
                dropout=0.0,
                target_modules=tuple(_LORA_SHAPE["target_modules"]),
            ),
            registry_sha256=kwargs.get("registry_sha256_override") or "",
            key_count=kwargs.get("key_count", 0),
            synthesized=False,
            window_stamp="",
        )

    monkeypatch.setattr(
        "experiments.test20_smallN_cold_gate.build_manifest_for",
        MagicMock(side_effect=_fake_build_manifest_for),
    )

    monkeypatch.setattr(
        "experiments.test20_smallN_cold_gate.atomic_save_adapter",
        MagicMock(side_effect=_fake_atomic_save_adapter),
    )

    return SimpleNamespace(
        entries=entries, captured=captured, checkpoint_root=tmp_path / "donor_checkpoint"
    )


class TestBuildDonorCheckpointDerivesOwnBudget:
    """The donor's OWN training always derives its epoch/accum/lr-decay
    budget from ``budget_for(len(donor_entries))`` — never inherits an
    arm's ``--epochs``/``--accum``/``--lr-decay-steps`` override carried on
    *base_training_config* (including the new meta fields)."""

    def test_donor_training_config_uses_budget_for_not_the_arm_override(self, _donor_build_env):
        # The arm's own overrides -- num_epochs=999, accum=1, lr_decay=550 --
        # must NOT leak into the donor's training config.
        base_training_config = dataclasses.replace(
            TrainingConfig(),
            num_epochs=999,
            gradient_accumulation_steps=1,
            lr_decay_steps=550,
            max_seq_length=1024,
        )
        expected_epochs, expected_accum, expected_lr_decay = budget_for(
            len(_donor_build_env.entries)
        )

        model, slot, donor_summary = _build_donor_checkpoint(
            _stub_model(),
            MagicMock(),
            AdapterConfig(),
            base_training_config,
            _donor_build_env.checkpoint_root,
        )

        trained_cfg = _donor_build_env.captured["training_config"]
        assert trained_cfg.num_epochs == expected_epochs
        assert trained_cfg.gradient_accumulation_steps == expected_accum
        assert trained_cfg.lr_decay_steps == expected_lr_decay
        assert (slot / DONOR_META_FILENAME).is_file()

    def test_donor_meta_and_summary_record_accum_and_realized_steps(self, _donor_build_env):
        """donor_meta.json / donor_summary gain
        gradient_accumulation_steps and realized_optimizer_steps for NEW
        builds, and the realized count matches the derived expectation."""
        base_training_config = dataclasses.replace(TrainingConfig(), max_seq_length=1024)
        expected_epochs, expected_accum, _ = budget_for(len(_donor_build_env.entries))
        expected_steps = _expected_optimizer_steps(
            len(_donor_build_env.entries), expected_epochs, expected_accum, batch_size=1
        )

        model, slot, donor_summary = _build_donor_checkpoint(
            _stub_model(),
            MagicMock(),
            AdapterConfig(),
            base_training_config,
            _donor_build_env.checkpoint_root,
        )

        assert donor_summary["gradient_accumulation_steps"] == expected_accum
        assert donor_summary["realized_optimizer_steps"] == expected_steps
        # Build measurements live in this script's own provenance file...
        provenance = json.loads((slot / DONOR_BUILD_PROVENANCE_FILENAME).read_text())
        assert provenance["gradient_accumulation_steps"] == expected_accum
        assert provenance["realized_optimizer_steps"] == expected_steps
        # ...while donor_meta.json carries the production donor identity only.
        meta = json.loads((slot / DONOR_META_FILENAME).read_text())
        assert meta["weights_sha256"] == donor_summary["weights_sha256"]
        assert meta["recipe"]
        assert "gradient_accumulation_steps" not in meta


class TestReadDonorMetaBuildProvenance:
    """Donor identity and this script's build measurements are separate files,
    so a slot built by PRODUCTION is a valid donor that simply carries no build
    provenance — reported as null, never fabricated."""

    def test_build_fields_come_from_the_provenance_file(self, tmp_path):
        slot = _write_slot(tmp_path, "own_build")

        result = _read_donor_meta(slot, *_VERIFY)

        assert result["seed"] == 42
        assert result["epochs"] == 30
        assert result["gradient_accumulation_steps"] == 2
        assert result["realized_optimizer_steps"] == 2220
        assert result["wall_train_seconds"] == 123.4

    def test_production_built_slot_is_valid_with_null_build_fields(self, tmp_path, monkeypatch):
        # caplog is unreliable in this environment (a third-party pytest
        # plugin reconfigures the logging module during collection), so
        # the info-log assertion patches the module's logger directly.
        info_mock = MagicMock()
        monkeypatch.setattr("experiments.test20_smallN_cold_gate.logger.info", info_mock)

        slot = _write_slot(tmp_path, "production_slot", provenance=False)

        result = _read_donor_meta(slot, *_VERIFY)

        assert result["seed"] == 42
        assert result["n_entries"] == 147
        assert result["epochs"] is None
        assert result["realized_optimizer_steps"] is None
        assert info_mock.called
        assert "provenance" in info_mock.call_args[0][0]


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
                "weights_sha256": "abc123",
                "wall_train_seconds": 456.7,
            }
        )
        monkeypatch.setattr("experiments.test20_smallN_cold_gate._read_donor_meta", read_meta_mock)
        # Defensive mocks for the seed-phase collaborators -- must NEVER be
        # invoked once the seed marker short-circuits the function; kept
        # here so a regression that moves the marker check surfaces as a
        # clean assertion failure rather than a real adapter-load attempt.
        load_adapter_mock = MagicMock()
        monkeypatch.setattr(
            "experiments.test20_smallN_cold_gate.PeftModel.from_pretrained", load_adapter_mock
        )
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
        load_adapter_mock.assert_not_called()
        copy_weights_mock.assert_not_called()
        # The build-phase results ARE recorded (build_results.json did not
        # pre-exist), proving only the SEED phase was skipped.
        assert (run_dir / "build_results.json").is_file()
